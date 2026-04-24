"""
**Tune — hybrid updatable stack focused on ``first_round_ration`` schedules.**

Isolates how the async/sync hand-off ratio interacts with updatable group engines; feeds choices into
hybrid updatable eval defaults.
"""

import copy
import gc
from typing import Any, Dict, List, Literal

import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import ConsensusExperimentBase
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import (
    RecommendationEngineGroupAllIndividualEaserUpdatable,
    RecommendationEngineGroupAllSameEaser,
)
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import (
    ConsensusMediatorHybridApproachWithFeedback,
    ThresholdPolicySigmoid,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    HybridMediatorUpdatableFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedProfilesUpdatable,
    GR_AggregatedRecommendations,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


class TuneHybridUpdatableByFirstRoundRation(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
    DEFAULT_AGG_STRATEGY = "mean"
    DEFAULT_POPULATION_BIASES = [0]
    DEFAULT_NDCG_KS = [20]
    DEFAULT_SIGMOID_PARAMS = {
        10: dict(
            center=[1],
            steepness=[0.75],
            c_init=[0.2],
            max_fill=[1],
            min_fill=0,
            first_r_ration=[2, 3, 4, 5, 6, 7],
        ),
        5: dict(
            center=[2],
            steepness=[0.75],
            c_init=[0.2],
            max_fill=[1],
            min_fill=0,
            first_r_ration=[2, 3, 4, 5],
        ),
        3: dict(
            center=[1],
            steepness=[0.2],
            c_init=[0.2],
            max_fill=[3],
            min_fill=0,
            first_r_ration=[1, 2, 3],
        ),
        1: dict(
            center=[1],
            steepness=[0.2],
            c_init=[0.2],
            max_fill=[1],
            min_fill=0,
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
            copy.deepcopy(sigmoid_params)
            if sigmoid_params is not None
            else copy.deepcopy(dict(type(self).DEFAULT_SIGMOID_PARAMS))
        )

    def _p(self) -> dict:
        if self.w_size not in self.sigmoid_params:
            raise ValueError(f"No SIGMOID_PARAMS for W_SIZE={self.w_size}")
        return self.sigmoid_params[self.w_size]

    def compute_results(self) -> Dict[str, Dict[Any, Any]]:
        p = self._p()
        first_round_l = list(p["first_r_ration"])
        center = p["center"][0]
        steepness = p["steepness"][0]
        c_init = p["c_init"][0]
        max_filling = p["max_fill"][0]
        min_filling = p["min_fill"]

        results: Dict[str, Dict[Any, Any]] = {gt: {} for gt in self.group_types}
        r = Runner()
        for group_type in self.group_types:
            r.refresh_context(self.eval_type, group_type)
            for first_round_r in first_round_l:
                factory_method = lambda single_user_model, evaluation_set_csr, w=self.w_size, fr=first_round_r: (
                    HybridMediatorUpdatableFactoryBuilder()
                    .with_updatable_group_rec_model(
                        lambda: GR_AggregatedProfilesUpdatable(single_user_model, update_mode="ema")
                    )
                    .with_general_recommender_engine_updatable(
                        lambda group, updatable_gr_rec_model: RecommendationEngineGroupAllIndividualEaserUpdatable(
                            group, updatable_group_model=updatable_gr_rec_model
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
                        lambda group, gen_updatable, gre, ru, th, w_=w, fr_=fr: ConsensusMediatorHybridApproachWithFeedback(
                            users_ids=group,
                            updatable_group_recommender=gen_updatable,
                            group_recommendation_engine=gre,
                            first_round_ration=fr_,
                            redistribution_unit=ru,
                            threshold_policy=th,
                            window_size=w_,
                        )
                    )
                    .build()
                )
                res = r.run(
                    factory_method,
                    self.groups_count,
                    self.ndcg_ks,
                    self.population_biases,
                    window_size=self.w_size,
                )
                results[group_type][first_round_r] = res
                gc.collect()
        return results

    def make_table(self, results: Dict[str, Dict[Any, Any]]) -> str:
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
            caption=rf"Tuning H1 (updatable) přes first\_r\_ration při fixních sigmoid parametrech, $w={self.w_size}$.",
            label="tab:h1_first_round_ration_grid",
            cell_bold_fn=lambda row_idx, col_idx, val: (
                col_idx >= 1 and pd.notna(val) and val == df.iloc[:, col_idx].min(skipna=True)
            ),
        )


if __name__ == "__main__":
    autorun(TuneHybridUpdatableByFirstRoundRation)

