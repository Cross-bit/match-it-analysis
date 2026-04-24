import gc
from typing import Any, Dict, List, Literal

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    latex_rfc_table_group_types_by_biases,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender import RecommendationEngineIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ThresholdPolicySigmoid
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    AsyncMediatorFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import RunnerLargeGroups
class EvalLargeAsyncWithSigmoidPolicySimplePriorityIndividualRec(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "test"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 1000
    DEFAULT_GROUP_SIZE = 10

    DEFAULT_GROUP_TYPES: List[str] = ["random"]
    DEFAULT_POPULATION_BIASES = [0]
    DEFAULT_NDCG_KS = [5, 10, 20, 50, 100]
    # Zarovnáno s eval_async_with_sigmoid_policy_simple_priority_individual_rec (core).
    DEFAULT_SIGMOID_PARAMS = {
        10: dict(center=1, steepness=0.75, c_init=0.2, max_fill=1, min_fill=0),
        5: dict(center=2, steepness=0.75, c_init=0.2, max_fill=1, min_fill=0),
        3: dict(center=1, steepness=0.2, c_init=0.2, max_fill=3, min_fill=0),
        1: dict(center=1, steepness=0.2, c_init=0.4, max_fill=1, min_fill=0),
    }

    def __init__(
        self,
        *,
        group_types: List[str] = None,
        population_biases: List[float] = None,
        ndcg_ks: List[int] = None,
        sigmoid_params: Dict[int, dict] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.population_biases = (
            population_biases if population_biases is not None else list(type(self).DEFAULT_POPULATION_BIASES)
        )
        self.ndcg_ks = ndcg_ks if ndcg_ks is not None else list(type(self).DEFAULT_NDCG_KS)
        self.sigmoid_params = (
            sigmoid_params if sigmoid_params is not None else dict(type(self).DEFAULT_SIGMOID_PARAMS)
        )

    def _sigmoid_row(self) -> Dict[str, Any]:
        if self.w_size not in self.sigmoid_params:
            raise ValueError(f"No SIGMOID_PARAMS for W_SIZE={self.w_size}")
        p = self.sigmoid_params[self.w_size]
        return {
            "w_size": self.w_size,
            "groups_count": self.groups_count,
            "center": p["center"],
            "steepness": p["steepness"],
            "c_init": p["c_init"],
            "max_fill": p["max_fill"],
            "min_fill": p["min_fill"],
        }

    def compute_results(self) -> Dict[str, Dict[str, Any]]:
        cfg = self._sigmoid_row()
        w_size = cfg["w_size"]
        groups_count = cfg["groups_count"]
        center = cfg["center"]
        steepness = cfg["steepness"]
        c_init = cfg["c_init"]
        max_fill = cfg["max_fill"]
        min_fill = cfg["min_fill"]

        results: Dict[str, Dict[str, Any]] = {}
        r = RunnerLargeGroups()
        for slot_i, _group_type in enumerate(self.group_types, start=1):
            self.cons_eval_set_progress_slot(slot_i)
            r.refresh_context(self.eval_type, self.group_size)

            def factory_method(single_user_model, evaluation_set_csr, w=w_size):
                return (
                    AsyncMediatorFactoryBuilder()
                    .with_recommender_engine(
                        lambda group: RecommendationEngineIndividualEaser(
                            group, model_iterator=single_user_model
                        )
                    )
                    .with_priority_function(
                        lambda group, model=single_user_model: SimplePriorityFunction(group, algorithm=model)
                    )
                    .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
                    .with_sigmoid_policy(
                        lambda ru,
                        center_=center,
                        steepness_=steepness,
                        c_init_=c_init,
                        max_fill_=max_fill,
                        min_fill_=min_fill: ThresholdPolicySigmoid(
                            red_context=getattr(ru, "redistribution_context", ru),
                            window_size=w,
                            sigmoid_center=center_,
                            sigmoid_steepness=steepness_,
                            c_init=c_init_,
                            max_filling=max_fill_,
                            min_filling=min_fill_,
                        )
                    )
                    .with_mediator(
                        lambda group, rec, ru, th, w_=w: ConsensusMediatorAsyncApproach(
                            group, rec, ru, th, window_size=w_
                        )
                    )
                    .build()
                )

            res = r.run(
                factory_method,
                groups_count,
                self.ndcg_ks,
                self.population_biases,
                window_size=w_size,
            )
            results[_group_type] = res
            gc.collect()

        return results

    def make_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        cfg = self._sigmoid_row()
        center = cfg["center"]
        steepness = cfg["steepness"]
        c_init = cfg["c_init"]
        max_fill = cfg["max_fill"]
        min_fill = cfg["min_fill"]
        w_size = cfg["w_size"]

        cap = (
            rf"RFC podle typu skupiny a population bias; velké skupiny ($n={self.group_size}$), "
            rf"async sigmoid individual rec, $(center={center}, steepness={steepness}, c\_init={c_init}, "
            rf"max={max_fill}, min={min_fill})$, $w={w_size}$."
        )
        return latex_rfc_table_group_types_by_biases(
            results=results,
            group_types=self.group_types,
            metric_key="average",
            caption=cap,
            label="tab:large_async_sigmoid_individual_rfc_by_bias",
        )


if __name__ == "__main__":
    autorun(EvalLargeAsyncWithSigmoidPolicySimplePriorityIndividualRec)
