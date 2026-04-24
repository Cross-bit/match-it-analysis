import gc
from typing import Any, Dict, List, Literal

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    latex_rfc_table_group_types_by_biases,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender import (
    RecommendationEngineGroupAllSameEaser,
    RecommendationEngineIndividualEaser,
)
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorHybridApproach, ThresholdPolicySigmoid
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    HybridMediatorFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import RunnerLargeGroups
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedRecommendations,
)


class EvalLargeHybridGeneralRecIndividual(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "test"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 1000
    DEFAULT_GROUP_SIZE = 10

    DEFAULT_GROUP_TYPES: List[str] = ["random"]
    DEFAULT_AGG_STRATEGY = "mean"
    DEFAULT_POPULATION_BIASES = [0]
    DEFAULT_NDCG_KS = [5, 10, 20, 50, 100]
    # Zarovnáno s eval_hybrid_general_rec_individual (core).
    DEFAULT_SIGMOID_PARAMS = {
        10: dict(center=1, steepness=0.75, c_init=0.2, max_fill=1, min_fill=0, first_r_ration=7),
        5: dict(center=2, steepness=0.75, c_init=0.2, max_fill=1, min_fill=0, first_r_ration=4),
        3: dict(center=1, steepness=0.2, c_init=0.2, max_fill=3, min_fill=0, first_r_ration=2),
        1: dict(center=1, steepness=0.2, c_init=0.2, max_fill=1, min_fill=0, first_r_ration=1),
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

    def _p(self) -> dict:
        if self.w_size not in self.sigmoid_params:
            raise ValueError(f"No SIGMOID_PARAMS for W_SIZE={self.w_size}")
        return self.sigmoid_params[self.w_size]

    def compute_results(self) -> Dict[str, Dict[str, Any]]:
        p = self._p()
        first_round_ration = p["first_r_ration"]
        center = p["center"]
        steepness = p["steepness"]
        c_init = p["c_init"]
        max_filling = p["max_fill"]
        min_filling = p["min_fill"]

        results: Dict[str, Dict[str, Any]] = {}
        r = RunnerLargeGroups()
        for slot_i, group_type in enumerate(self.group_types, start=1):
            self.cons_eval_set_progress_slot(slot_i)
            r.refresh_context(self.eval_type, self.group_size)
            factory_method = lambda single_user_model, evaluation_set_csr, w=self.w_size, fr=first_round_ration: (
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
            res = r.run(
                factory_method,
                self.groups_count,
                self.ndcg_ks,
                self.population_biases,
                window_size=self.w_size,
            )
            results[group_type] = res
            gc.collect()
        return results

    def make_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        p = self._p()
        center = p["center"]
        steepness = p["steepness"]
        c_init = p["c_init"]
        max_filling = p["max_fill"]
        min_filling = p["min_fill"]

        cap = (
            rf"RFC podle typu skupiny a population bias; velké skupiny ($n={self.group_size}$), "
            rf"hybrid (general rec individual), "
            rf"sigmoid $(center={center}, steepness={steepness}, c\_init={c_init}, "
            rf"max={max_filling}, min={min_filling})$, $w={self.w_size}$."
        )
        return latex_rfc_table_group_types_by_biases(
            results=results,
            group_types=self.group_types,
            metric_key="average",
            caption=cap,
            label="tab:large_hybrid_general_rec_rfc_by_bias",
        )


if __name__ == "__main__":
    autorun(EvalLargeHybridGeneralRecIndividual)
