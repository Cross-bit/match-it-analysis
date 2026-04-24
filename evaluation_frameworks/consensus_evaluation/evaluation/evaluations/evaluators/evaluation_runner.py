"""
``Runner`` — high-level **batch driver** for one evaluation split × group type.

Workflow: ``refresh_context(eval_split, group_type)`` loads tensors/maps via
``evaluation_context_factory``, then ``run(factory_builder, ...)`` loops over groups, builds each
mediator through the supplied **factory** (from ``consensus_mediator_factories``), and delegates
per-group simulation to ``ConsensusAgentBasedEvaluator``. Handles population-mood biases, optional
NDCG@k, worker pools (threads or optional Linux fork pool), and progress hooks.

This module also hosts small helpers such as ``resolve_simulation_max_rounds`` used by evaluators.
"""

from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
from lightfm import LightFM
from time import time

import numpy as np
from tqdm import tqdm
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluation_context_factory import build_context, build_context_holdout, build_context_large_holdout, load_evaluation_agent_base, load_evaluation_agent_sigmoid_normed
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_evaluator import ConsensusAgentBasedEvaluator
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import AsyncMediatorFactoryBuilder
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_data_interpreter import print_evaluation_result
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineGroupAllIndividualEaserUpdatable
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ThresholdPolicyStatic
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import GR_AggregatedRecommendations
from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.model_train_load import train_or_load_easer_model
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserSparse
from utils.config import load_from_pickle, load_or_build_pickle
from scipy.sparse import csr_matrix
import gc
import os
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.batch_run_progress import (
    print_bias_completed_global_progress,
    print_runner_batch_preamble,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.debug_profile import log_event, timed


def resolve_simulation_max_rounds(window_size: Optional[int] = None) -> int:
    """
    Max consensus rounds per group before the run counts as "no first match".

    Priority:
      1. ``CONS_EVAL_MAX_ROUNDS`` — explicit override.
      2. If ``window_size`` is set: ``ceil(CONS_EVAL_ROUND_BUDGET / W)``
         (default budget 100 → keeps ``W * max_rounds`` ≈ 100 slot-steps).
      3. Legacy default: 10 (when ``window_size`` is unknown).
    """
    env_mr = os.environ.get("CONS_EVAL_MAX_ROUNDS")
    if env_mr is not None and str(env_mr).strip() != "":
        return max(1, int(env_mr))
    if window_size is None:
        return 10
    w = max(1, int(window_size))
    budget = int(os.environ.get("CONS_EVAL_ROUND_BUDGET", "100"))
    return max(1, (budget + w - 1) // w)


class Runner:
    """Orchestrates cached data context + repeated group-level consensus simulations."""

    def __init__(self,
                cache_identifier="hold-out",
                ignore_cache=False):
        """Prepare an empty runner; call ``refresh_context`` before ``run``."""

        self.EVALUATION_SET_TYPE = None
        self.GROUP_TYPE = None

        self.context = None
        self.evaluation_set_csr = None
        self.user_id_map = None
        self.item_id_map = None

        self.ignore_cache = ignore_cache
        self.cache_identifier = cache_identifier

        self.single_user_model: EaserSparse = None

        self.simulation_agent = None

    # 1.
    #  FIRST CALL THIS!!!!:
    #
    def refresh_context(self, EVALUATION_SET_TYPE: Literal["train", "validation", "test"],
                                GROUP_TYPE: Literal["similar", "outlier", "random", "divergent", "variance"]):
        with timed("runner.refresh_context", extra={"eval_type": EVALUATION_SET_TYPE, "group_type": GROUP_TYPE}):
            self.context = build_context_holdout(EVALUATION_SET_TYPE, GROUP_TYPE)

        self.evaluation_set_csr = self.context["filtered_evaluation_set_csr"]
        self.user_id_map = self.context["user_id_map"]
        self.item_id_map = self.context["item_id_map"]

        self.EVALUATION_SET_TYPE = EVALUATION_SET_TYPE
        self.GROUP_TYPE = GROUP_TYPE

    # 2.
    # THEN CALL THIS!!!!:
    #
    def run(self, factory_build_method: Callable[[Any, Any], Any],
            evaluation_size: int = 1000,
            ndcg_k: Optional[List[int]] = None,
            selection_biases: List[int] = [0], # todo: this actually may be done bit more optimal... but lets leave it this way...
            end_on_first_match = False, # NOTE: this is actually important so the NDCG for larger numbers is computed properly ...
            *,
            window_size: Optional[int] = None,
            max_rounds: Optional[int] = None,
            ):
        """evaluation_size = number of groups to simulate (CLI: --groups-count on experiments)."""

        if self.context == None:
            raise Exception("refresh_context must be called first!!")

        eval_set_groups_data = self.context["eval_set_groups_data"]

        self._ensure_single_user_rec()
        log_event("runner.context.ready", extra={"groups_available": len(eval_set_groups_data), "evaluation_size": evaluation_size})

        result = {}
        workers = int(os.environ.get("CONS_EVAL_WORKERS", "6"))
        log_event("runner.workers", extra={"workers": workers})

        n_bias = len(selection_biases)
        print(
            "\n"
            + "=" * 72
            + "\n"
            + "Runner — population mood biases (pořadí v tomto běhu)\n"
            + f"  biases ({n_bias}): {selection_biases!r}\n"
            + f"  eval_split={self.EVALUATION_SET_TYPE!r}  group_type={self.GROUP_TYPE!r}\n"
            + f"  groups_count={evaluation_size}  CONS_EVAL_WORKERS={workers}\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
        print_runner_batch_preamble(self.GROUP_TYPE, n_bias)

        _preview_mr = (
            max(1, int(max_rounds))
            if max_rounds is not None
            else resolve_simulation_max_rounds(window_size)
        )
        print(f"[runner] simulation max_rounds={_preview_mr}", flush=True)

        for b_idx, selection_bias in enumerate(selection_biases, start=1):
            print(
                f"\n>>> Bias {b_idx}/{n_bias}: population_mood_bias = {selection_bias!r} "
                "(načtení agenta + simulace skupin + NDCG pro tento bias)\n",
                flush=True,
            )
            log_event(
                "runner.bias.start",
                extra={
                    "bias_index": b_idx,
                    "n_biases": n_bias,
                    "population_mood_bias": selection_bias,
                    "group_type": self.GROUP_TYPE,
                    "eval_type": self.EVALUATION_SET_TYPE,
                },
            )

            with timed("runner.factory.build", extra={"selection_bias": selection_bias}):
                factory = factory_build_method(self.single_user_model, self.evaluation_set_csr)

            with timed("runner.agent.load", extra={"selection_bias": selection_bias}):
                self._ensure_simulation_agent(selection_bias)

            sim_max_rounds = (
                max(1, int(max_rounds))
                if max_rounds is not None
                else resolve_simulation_max_rounds(window_size)
            )
            log_event(
                "runner.simulation_max_rounds",
                extra={"max_rounds": sim_max_rounds, "window_size": window_size},
            )

            evaluator = ConsensusAgentBasedEvaluator(
                self.simulation_agent,
                self.EVALUATION_SET_TYPE,
                eval_set_groups_data,
                self.context["groups_ground_truth"],
                self.context["users_ground_truth"],
                self.GROUP_TYPE,
                max_rounds=sim_max_rounds,
                end_on_first_match = end_on_first_match
            )

            gc.collect()
            with timed("runner.simulation", extra={"selection_bias": selection_bias, "evaluation_size": evaluation_size}):
                res = evaluator.run_simulation(factory, evaluation_size, ndcg_k, workers=workers)

            result[selection_bias] = res
            print_bias_completed_global_progress(self.GROUP_TYPE, b_idx, n_bias)

        return result

    def _ensure_simulation_agent(self, global_user_bias: float):
        self.simulation_agent = load_evaluation_agent_sigmoid_normed(
            self.evaluation_set_csr, self.user_id_map, self.item_id_map,
            self.EVALUATION_SET_TYPE, identifier=self.cache_identifier,
            global_user_bias=global_user_bias,
            force_rebuild_cache=self.ignore_cache
        )

    def _ensure_single_user_rec(self):
        if not self.single_user_model:
            self.easer_model_name = f"{self.EVALUATION_SET_TYPE}-groups-evaluation-model-easer-{self.cache_identifier}.pkl"
            with timed("runner.model.easer.load_or_train", extra={"model_name": self.easer_model_name}):
                self.single_user_model = train_or_load_easer_model(
                    self.evaluation_set_csr, self.user_id_map, self.item_id_map,
                    model_name=self.easer_model_name
                )


class RunnerLargeGroups:
    """Like ``Runner`` but loads **large-group** holdouts via ``build_context_large_holdout`` (fixed ``group_size``)."""

    def __init__(self,
                cache_identifier="hold-out",
                ignore_cache=False):
        """Prepare empty large-group runner; set ``group_size`` before ``refresh_context`` / ``run``."""

        self.EVALUATION_SET_TYPE = None

        self.context = None
        self.evaluation_set_csr = None
        self.user_id_map = None
        self.item_id_map = None

        self.ignore_cache = ignore_cache
        self.cache_identifier = cache_identifier

        self.single_user_model: EaserSparse = None

        self.simulation_agent = None

        self.group_size = 3


    def refresh_context(self, EVALUATION_SET_TYPE: Literal["train", "validation", "test"], group_size):
        with timed("runner_large.refresh_context", extra={"eval_type": EVALUATION_SET_TYPE, "group_size": group_size}):
            self.context = build_context_large_holdout(EVALUATION_SET_TYPE, "random", group_size)

        self.evaluation_set_csr = self.context["filtered_evaluation_set_csr"]
        self.user_id_map = self.context["user_id_map"]
        self.item_id_map = self.context["item_id_map"]

        self.EVALUATION_SET_TYPE = EVALUATION_SET_TYPE

    def run(self, factory_build_method: Callable[[Any, Any], Any],
            evaluation_size: int = 1000,
            ndcg_k: Optional[List[int]] = None,
            selection_biases: List[int] = [0], # todo: this actually may be done bit more optimal... but lets leave it this way...
            *,
            window_size: Optional[int] = None,
            max_rounds: Optional[int] = None,
            ):
        """evaluation_size = number of groups to simulate (CLI: --groups-count on experiments)."""

        if self.context == None:
            raise Exception("refresh_context must be called first!!")

        eval_set_groups_data = self.context["eval_set_groups_data"]

        self._ensure_single_user_rec()

        result = {}
        workers = int(os.environ.get("CONS_EVAL_WORKERS", "6"))
        log_event("runner_large.workers", extra={"workers": workers})

        n_bias = len(selection_biases)
        print(
            "\n"
            + "=" * 72
            + "\n"
            + "RunnerLargeGroups — population mood biases (pořadí v tomto běhu)\n"
            + f"  biases ({n_bias}): {selection_biases!r}\n"
            + f"  eval_split={self.EVALUATION_SET_TYPE!r}  group_type=random (large)\n"
            + f"  groups_count={evaluation_size}  CONS_EVAL_WORKERS={workers}\n"
            + "=" * 72
            + "\n",
            flush=True,
        )
        print_runner_batch_preamble("random", n_bias)

        _preview_mr_lg = (
            max(1, int(max_rounds))
            if max_rounds is not None
            else resolve_simulation_max_rounds(window_size)
        )
        print(f"[runner_large] simulation max_rounds={_preview_mr_lg}", flush=True)

        for b_idx, selection_bias in enumerate(selection_biases, start=1):
            print(
                f"\n>>> Bias {b_idx}/{n_bias}: population_mood_bias = {selection_bias!r} "
                "(načtení agenta + simulace skupin + NDCG pro tento bias)\n",
                flush=True,
            )
            log_event(
                "runner_large.bias.start",
                extra={
                    "bias_index": b_idx,
                    "n_biases": n_bias,
                    "population_mood_bias": selection_bias,
                    "eval_type": self.EVALUATION_SET_TYPE,
                },
            )

            with timed("runner_large.factory.build", extra={"selection_bias": selection_bias}):
                factory = factory_build_method(self.single_user_model, self.evaluation_set_csr)

            with timed("runner_large.agent.load", extra={"selection_bias": selection_bias}):
                self._ensure_simulation_agent(selection_bias)

            sim_max_rounds = (
                max(1, int(max_rounds))
                if max_rounds is not None
                else resolve_simulation_max_rounds(window_size)
            )
            log_event(
                "runner_large.simulation_max_rounds",
                extra={"max_rounds": sim_max_rounds, "window_size": window_size},
            )

            evaluator = ConsensusAgentBasedEvaluator(
                self.simulation_agent,
                self.EVALUATION_SET_TYPE,
                eval_set_groups_data,
                self.context["groups_ground_truth"],
                self.context["users_ground_truth"],
                "random",
                max_rounds=sim_max_rounds,
                end_on_first_match = False # NOTE: this is actually important so the NDCG for larger numbers is computed properly ...
            )

            gc.collect()
            with timed("runner_large.simulation", extra={"selection_bias": selection_bias, "evaluation_size": evaluation_size}):
                res = evaluator.run_simulation(factory, evaluation_size, ndcg_k, workers=workers)

            result[selection_bias] = res
            print_bias_completed_global_progress("random", b_idx, n_bias)

        return result

    def _ensure_simulation_agent(self, global_user_bias: float):
        self.simulation_agent = load_evaluation_agent_sigmoid_normed(
            self.evaluation_set_csr, self.user_id_map, self.item_id_map,
            self.EVALUATION_SET_TYPE, identifier=self.cache_identifier,
            global_user_bias=global_user_bias,
            force_rebuild_cache=self.ignore_cache
        )

    def _ensure_single_user_rec(self):
        if not self.single_user_model:
            self.easer_model_name = f"{self.EVALUATION_SET_TYPE}-groups-evaluation-model-easer-{self.cache_identifier}.pkl"
            with timed("runner_large.model.easer.load_or_train", extra={"model_name": self.easer_model_name}):
                self.single_user_model = train_or_load_easer_model(
                    self.evaluation_set_csr, self.user_id_map, self.item_id_map,
                    model_name=self.easer_model_name
                )