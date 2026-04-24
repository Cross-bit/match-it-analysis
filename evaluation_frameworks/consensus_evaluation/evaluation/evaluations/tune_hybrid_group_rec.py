"""
**Tune — hybrid mediator emphasizing the group-recommender branch hyperparameters.**

Sweeps knobs tied to ``RecommendationEngineGroupAllIndividualEaser`` / hybrid group path; complements
``tune_hybrid_general_rec_individual.py``.
"""

import gc
from typing import Any, Dict, List, Literal

import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import ConsensusExperimentBase
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import (
    RecommendationEngineGroupAllIndividualEaser,
    RecommendationEngineGroupAllSameEaser,
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


class TuneHybridGroupRec(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
    DEFAULT_AGG_STRATEGY = "mean"
    DEFAULT_POPULATION_BIASES = [0]
    DEFAULT_SIGMOID_PARAMS = {
        10: dict(
            center=[1],
            steepness=[0.2],
            c_init=[0, 0.2, 0.4, 0.7],
            max_fill=[1, 2, 5, 10, 15, 20],
            min_fill=0,
            first_r_ration=[2, 3, 4],
        ),
        5: dict(
            center=[1, 2, 3],
            steepness=[0.2, 0.75, 1.4],
            c_init=[0, 0.2, 0.4, 0.7],
            max_fill=[1, 2, 5, 10, 15, 20],
            min_fill=0,
            first_r_ration=[1, 2, 3, 4, 5],
        ),
        1: dict(
            center=[1],
            steepness=[0.2],
            c_init=[0, 0.2, 0.4],
            max_fill=[1],
            min_fill=0,
            first_r_ration=[1, 2, 3],
        ),
    }

    FIXED_CENTER = 1
    FIXED_STEEPNESS = 0.75
    FIXED_C_INIT = 0.2
    FIXED_MAX_FILLING = 5
    FIXED_MIN_FILLING = 0

    def __init__(
        self,
        *,
        group_types: List[str] = None,
        agg_strategy: str = None,
        population_biases: List[float] = None,
        sigmoid_params: Dict[int, dict] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.agg_strategy = agg_strategy if agg_strategy is not None else type(self).DEFAULT_AGG_STRATEGY
        self.population_biases = (
            population_biases if population_biases is not None else list(type(self).DEFAULT_POPULATION_BIASES)
        )
        self.sigmoid_params = (
            sigmoid_params if sigmoid_params is not None else dict(type(self).DEFAULT_SIGMOID_PARAMS)
        )

    def _first_round_list(self) -> List[int]:
        if self.w_size not in self.sigmoid_params:
            raise ValueError(f"No SIGMOID_PARAMS for W_SIZE={self.w_size}")
        return self.sigmoid_params[self.w_size]["first_r_ration"]

    def compute_results(self) -> Dict[str, Dict[Any, Any]]:
        first_round_l = self._first_round_list()
        results: Dict[str, Dict[Any, Any]] = {gt: {} for gt in self.group_types}
        center = self.FIXED_CENTER
        steepness = self.FIXED_STEEPNESS
        c_init = self.FIXED_C_INIT
        max_filling = self.FIXED_MAX_FILLING
        min_filling = self.FIXED_MIN_FILLING

        r = Runner()
        for group_type in self.group_types:
            r.refresh_context(self.eval_type, group_type)
            for first_round_r in first_round_l:
                factory_method = lambda single_user_model, evaluation_set_csr, w=self.w_size, fr=first_round_r: (
                    HybridMediatorFactoryBuilder()
                    .with_general_recommender_engine(
                        lambda group, model=single_user_model, csr=evaluation_set_csr: RecommendationEngineGroupAllIndividualEaser(
                            group, model=GR_AggregatedRecommendations(model)
                        )
                    )
                    .with_priority_function(
                        lambda group, model=single_user_model: SimplePriorityFunction(group, algorithm=model)
                    )
                    .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
                    .with_sigmoid_policy(
                        lambda ru: ThresholdPolicySigmoid(
                            red_context=getattr(ru, "redistribution_context", ru),
                            window_size=w,
                            sigmoid_center=center,
                            sigmoid_steepness=steepness,
                            c_init=c_init,
                            max_filling=max_filling,
                            min_filling=min_filling,
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
                res = r.run(factory_method, self.groups_count, window_size=self.w_size)
                results[group_type][first_round_r] = res
                gc.collect()
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
    autorun(TuneHybridGroupRec)
