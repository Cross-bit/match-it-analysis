"""
**Tune — sync + feedback (EMA-style aggregation strategies / alphas).**

Uses process pools for some sweeps; picks feedback hyperparameters for the sync-with-feedback family
that ``eval_sync_with_feedback_ema`` can fix for fixed-parameter eval runs.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Tuple

import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import ConsensusExperimentBase
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineGroupAllSameEaserWithFeedback
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorSyncApproach
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    SyncMediatorFactoryBuilderSync,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedProfilesUpdatable,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


def _evaluate_group_type(payload: Tuple[str, str, int, int, List[str], List[float]]) -> Tuple[str, dict]:
    group_type, eval_type, groups_count, w_size, agg_strategies, alpha_values = payload
    r = Runner()
    r.refresh_context(eval_type, group_type)
    partial_results = {}
    for agg_strategy in agg_strategies:
        for alpha_value in alpha_values:
            av = alpha_value
            strat = agg_strategy
            factory_method = lambda single_user_model, evaluation_set_csr, alpha_=av, s=strat: (
                SyncMediatorFactoryBuilderSync()
                .with_group_algorithm(
                    lambda: GR_AggregatedProfilesUpdatable(
                        single_user_model, update_mode="ema", alpha=alpha_
                    )
                )
                .with_group_recommender_engine(
                    lambda group, eg: RecommendationEngineGroupAllSameEaserWithFeedback(group, eg, s)
                )
                .with_mediator(
                    lambda group, gre: ConsensusMediatorSyncApproach(group, w_size, gre)
                )
                .build()
            )
            res = r.run(factory_method, groups_count, window_size=w_size)
            partial_results[(agg_strategy, alpha_value)] = res
    return group_type, partial_results


class TuneSyncWithFeedbackEma(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_AGG_STRATEGIES: List[str] = ["mean", "min", "max", "median", "plurality"]
    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
    DEFAULT_ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.7]

    def __init__(
        self,
        *,
        agg_strategies: List[str] = None,
        group_types: List[str] = None,
        alpha_values: List[float] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.agg_strategies = (
            agg_strategies if agg_strategies is not None else list(type(self).DEFAULT_AGG_STRATEGIES)
        )
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.alpha_values = (
            alpha_values if alpha_values is not None else list(type(self).DEFAULT_ALPHA_VALUES)
        )

    def compute_results(self) -> dict:
        # Každý worker proces načte celý eval kontext + modely → 2× paralelně snadno OOM na WSL.
        # Výchozí 1; paralelizace: TUNE_EMA_POOL_WORKERS=2 (potřebuješ hodně RAM).
        pool_workers = max(1, min(int(os.environ.get("TUNE_EMA_POOL_WORKERS", "1")), len(self.group_types)))
        results = {}
        with ProcessPoolExecutor(max_workers=pool_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_group_type,
                    (
                        gt,
                        self.eval_type,
                        self.groups_count,
                        self.w_size,
                        self.agg_strategies,
                        self.alpha_values,
                    ),
                ): gt
                for gt in self.group_types
            }
            for future in as_completed(futures):
                group_type, partial_results = future.result()
                results[group_type] = partial_results
        return results

    def make_table(self, results: Dict[str, Dict[Any, Any]]) -> str:
        rows = []
        for gt in self.group_types:
            for key, metrics in results[gt].items():
                row = {"params": key, gt: metrics[0]["average"]}
                rows.append(row)
        df = pd.DataFrame(rows).groupby("params").first().reset_index()
        cols = ["params"] + [gt for gt in self.group_types if gt in df.columns]
        df = df[cols]
        if len(self.group_types) > 0:
            df["average"] = df[self.group_types].mean(axis=1)
        generator = LaTeXTableGeneratorSIUnitx(
            df,
            column_specs=[(1, 2)] * (len(cols) - 1) + [(1, 3)],
            column_width=1.5,
        )
        return generator.generate_table(
            caption=(
                r"Vyhodnocení metriky RFC pro synchronní algoritmus se zpětnou vazbou (EMA) "
                r"pro různé agregační strategie a hodnoty parametru $\alpha$. "
                r"Řádky odpovídají kombinacím (strategie, $\alpha$), výsledky jsou agregovány přes typy skupin."
            ),
            label="tab:tune_sync_strategy_with_feedback_mean",
            cell_bold_fn=lambda row_idx, col_idx, val: (
                col_idx >= 1 and pd.notna(val) and val == df.iloc[:, col_idx].min(skipna=True)
            ),
        )


if __name__ == "__main__":
    autorun(TuneSyncWithFeedbackEma)
