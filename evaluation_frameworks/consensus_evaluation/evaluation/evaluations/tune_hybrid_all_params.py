"""
**Tune — wide joint search across many hybrid hyperparameters (H0/H1 style).**

Intended for large exploratory sweeps rather than tight daily reruns; coordinates many
combinations (see ``run_consensus.sh tune-hybrid`` orchestration). Use smaller group counts by default.
"""

import gc
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from itertools import product
from typing import Any, Dict, List, Literal

import pandas as pd
from tqdm import tqdm

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import ConsensusExperimentBase
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import (
    RecommendationEngineGroupAllSameEaser,
    RecommendationEngineIndividualEaser,
)
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorHybridApproach, ThresholdPolicySigmoid
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    HybridMediatorFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedRecommendations,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


class _NullStdout:
    def write(self, s):
        return len(s)

    def flush(self):
        pass

    def close(self):
        pass


_SILENT_STDOUT = _NullStdout()


@contextmanager
def quiet_stdout(disable_logging_level=logging.CRITICAL, disable_warnings=True):
    prev_disable = logging.root.manager.disable
    logging.disable(disable_logging_level)

    warnings_ctx = None
    if disable_warnings:
        warnings_ctx = warnings.catch_warnings()
        warnings_ctx.__enter__()
        warnings.simplefilter("ignore")

    old_stdout = sys.stdout
    sys.stdout = _SILENT_STDOUT
    try:
        yield
    finally:
        sys.stdout = old_stdout
        logging.disable(prev_disable)
        if warnings_ctx is not None:
            warnings_ctx.__exit__(None, None, None)


class TuneHybridAllParams(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 10

    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
    DEFAULT_AGG_STRATEGY = "mean"
    DEFAULT_POPULATION_BIASES = [0]
    DEFAULT_NDCG_KS = [20]
    DEFAULT_SIGMOID_PARAMS = {
        10: dict(
            center=[1, 2, 5],
            steepness=[0.2, 0.75, 1.4],
            c_init=[0, 0.2, 0.4, 0.7],
            max_fill=[1, 2, 5, 10, 15, 20],
            min_fill=[0],
            first_r_ration=[2, 3, 4],
        ),
        5: dict(
            center=[1, 2, 3],
            steepness=[0.2, 0.75, 1.4],
            c_init=[0, 0.2, 0.4, 0.7],
            max_fill=[1, 2, 5, 10, 15, 20],
            min_fill=[0],
            first_r_ration=[1, 2, 3],
        ),
        3: dict(
            center=[1, 2, 3],
            steepness=[0.2, 0.75, 1.4],
            c_init=[0, 0.2, 0.4, 0.7],
            max_fill=[1, 2, 3],
            min_fill=[0],
            first_r_ration=[1, 2, 3],
        ),
        1: dict(
            center=[1, 2],
            steepness=[0.2, 0.75, 1.4],
            c_init=[0, 0.2, 0.4],
            max_fill=[1],
            min_fill=[0],
            first_r_ration=[1, 2, 3],
        ),
    }

    def __init__(
        self,
        *,
        group_types: List[str] = None,
        agg_strategy: str = None,
        population_biases: List[float] = None,
        ndcg_ks: List[int] = None,
        sigmoid_params: Dict[int, dict] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.agg_strategy = agg_strategy if agg_strategy is not None else type(self).DEFAULT_AGG_STRATEGY
        self.population_biases = (
            population_biases if population_biases is not None else list(type(self).DEFAULT_POPULATION_BIASES)
        )
        self.ndcg_ks = ndcg_ks if ndcg_ks is not None else list(type(self).DEFAULT_NDCG_KS)
        self.sigmoid_params = (
            sigmoid_params if sigmoid_params is not None else dict(type(self).DEFAULT_SIGMOID_PARAMS)
        )

    def _grid_axes(self):
        if self.w_size not in self.sigmoid_params:
            raise ValueError(f"No SIGMOID_PARAMS for W_SIZE={self.w_size}")
        p = self.sigmoid_params[self.w_size]
        raw_min = p["min_fill"]
        if isinstance(raw_min, (int, float)):
            min_fill_vals = [raw_min]
        else:
            min_fill_vals = list(raw_min)
        return (
            p["first_r_ration"],
            p["center"],
            p["steepness"],
            p["c_init"],
            p["max_fill"],
            min_fill_vals,
        )

    def compute_results(self) -> Dict[str, Dict[tuple, Dict[str, Any]]]:
        first_rr, centers, steepnesses, c_inits, max_fills, min_fill_vals = self._grid_axes()
        results: Dict[str, Dict[tuple, Dict[str, Any]]] = {gt: {} for gt in self.group_types}
        grid = list(product(first_rr, centers, steepnesses, c_inits, max_fills, min_fill_vals))

        def run_for_group_type(group_type: str):
            local_results = {}
            r = Runner()
            r.refresh_context(self.eval_type, group_type)
            start_time = time.time()

            with tqdm(grid, desc=f"Grid @{group_type}", leave=False, dynamic_ncols=True) as pbar:
                for first_round_r, center, steepness, c_init, max_fill, min_filling in pbar:
                    # Výchozí argumenty lambdy: vždycky svážou hodnoty z této iterace (jinak paralelní
                    # vlákna / pozdější iterace přepíšou closure a do sigmoidu vlézne třeba list místo skaláru).
                    factory_method = lambda single_user_model, evaluation_set_csr, w=self.w_size, fr=first_round_r, c_=center, st_=steepness, ci_=c_init, mf_=max_fill, minf=min_filling: (
                        HybridMediatorFactoryBuilder()
                        .with_general_recommender_engine(
                            lambda group: RecommendationEngineIndividualEaser(group, model_iterator=single_user_model)
                        )
                        .with_priority_function(
                            lambda group, model=single_user_model: SimplePriorityFunction(group, algorithm=model)
                        )
                        .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
                        .with_sigmoid_policy(
                            lambda ru: ThresholdPolicySigmoid(
                                red_context=getattr(ru, "redistribution_context", ru),
                                window_size=w,
                                sigmoid_center=c_,
                                sigmoid_steepness=st_,
                                c_init=ci_,
                                max_filling=mf_,
                                min_filling=minf,
                            )
                        )
                        .with_group_algorithm(lambda: GR_AggregatedRecommendations(single_user_model))
                        .with_group_recommender_engine(
                            lambda group, eg: RecommendationEngineGroupAllSameEaser(
                                group, evaluation_set_csr, eg, self.agg_strategy
                            )
                        )
                        .with_mediator(
                            lambda group, gen, gre, ru, th, w_=w, fr_=fr: ConsensusMediatorHybridApproach(
                                users_ids=group,
                                general_recommender=gen,
                                group_recommendation_engine=gre,
                                first_round_ration=fr_,
                                redistribution_unit=ru,
                                threshold_policy=th,
                                window_size=w_,
                            )
                        )
                        .build()
                    )

                    key = (first_round_r, c_init, center, steepness, max_fill, min_filling)

                    try:
                        with quiet_stdout():
                            res = r.run(
                                factory_method,
                                self.groups_count,
                                self.ndcg_ks,
                                self.population_biases,
                                window_size=self.w_size,
                            )
                        local_results[key] = res
                    except BaseException as e:
                        pbar.set_postfix_str(f"ERROR at {key}: {e.__class__.__name__}")
                        raise
                    finally:
                        gc.collect()
                        elapsed = time.time() - start_time
                        mins, secs = divmod(int(elapsed), 60)
                        pbar.set_postfix_str(f"elapsed: {mins:02d}:{secs:02d}")

            return group_type, local_results

        grouptype_workers = max(
            1,
            min(int(os.environ.get("TUNE_HYBRID_H0_GROUPTYPE_WORKERS", "2")), len(self.group_types)),
        )
        if grouptype_workers == 1:
            # Keep execution on main thread so run_simulation can enable process-pool mode.
            for gt in tqdm(self.group_types, total=len(self.group_types), desc="Group types", dynamic_ncols=True):
                key, partial = run_for_group_type(gt)
                results[key] = partial
        else:
            with ThreadPoolExecutor(max_workers=grouptype_workers) as ex:
                futs = [ex.submit(run_for_group_type, gt) for gt in self.group_types]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Group types", dynamic_ncols=True):
                    gt, partial = fut.result()
                    results[gt] = partial

        return results

    def make_table(self, results: Dict[str, Dict[tuple, Dict[str, Any]]]) -> str:
        rows = []
        for gt in self.group_types:
            for key, metrics in results[gt].items():
                rows.append({"params": key, gt: metrics[0]["average"]})
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
            caption=rf"Vyhodnocení metrik RFC pro asynchronní doporučovač se sigmoid policy. "
            r"Řádky odpovídají kombinacím hyperparametrů $(center, steepness, c\_init, max, min)$ "
            rf"při okně $w={self.w_size}$.",
            label="tab:async_sigmoid_threshold_grid",
            cell_bold_fn=lambda row_idx, col_idx, val: (
                col_idx >= 1 and pd.notna(val) and val == df.iloc[:, col_idx].min(skipna=True)
            ),
        )


if __name__ == "__main__":
    autorun(TuneHybridAllParams)
