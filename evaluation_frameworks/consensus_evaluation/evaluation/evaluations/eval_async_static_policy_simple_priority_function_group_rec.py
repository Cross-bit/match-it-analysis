"""
**Eval — async mediator, static threshold, group-level EASer recommender.**

Uses ``RecommendationEngineGroupAllIndividualEaser`` + aggregated group EASer: candidate lists come
from a **group** recommender path while still using async round structure and static window thresholds.
"""

import gc
from multiprocessing import get_context
from typing import Any, Dict, List, Literal

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    latex_rfc_table_group_types_by_biases,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineGroupAllIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ThresholdPolicyStatic
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    AsyncMediatorFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.batch_run_progress import (
    set_progress_slot_before_runner,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedRecommendations,
)
def _compute_one_group(args):
    import ctypes

    (
        slot_idx,
        n_slots,
        n_bias,
        group_type,
        eval_type,
        groups_count,
        w_size,
        t_value,
        ndcg_ks,
        population_biases,
    ) = args

    set_progress_slot_before_runner(slot_idx, n_slots, n_bias)
    r = Runner()
    r.refresh_context(eval_type, group_type)

    def factory_method(single_user_model, evaluation_set_csr, t=t_value, window_size=w_size):
        return (
            AsyncMediatorFactoryBuilder()
            .with_recommender_engine(
                lambda group, model=single_user_model, csr=evaluation_set_csr: RecommendationEngineGroupAllIndividualEaser(
                    group, model=GR_AggregatedRecommendations(model)
                )
            )
            .with_priority_function(
                lambda group, model=single_user_model: SimplePriorityFunction(group, algorithm=model)
            )
            .with_threshold_policy(lambda t_=t: ThresholdPolicyStatic(t_param=t_))
            .with_redistribution(lambda group, priority_fun: RedistributionUnit(group, priority_fun))
            .with_mediator(
                lambda group_ids, rec, ru, th, window_size_=window_size: ConsensusMediatorAsyncApproach(
                    group_ids, rec, ru, th, window_size=window_size_
                )
            )
            .build()
        )

    res = r.run(
        factory_method,
        groups_count,
        ndcg_ks,
        population_biases,
        window_size=w_size,
    )

    del r
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

    return group_type, res


class EvalAsyncStaticPolicySimplePriorityGroupRec(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "test"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 1000

    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
    DEFAULT_POPULATION_BIASES = [0, 0.5, 1, 2, 3, 5]
    DEFAULT_NDCG_KS = [5, 10, 20, 50, 100]
    DEFAULT_STATIC_PARAMS = {
        10: dict(t_value=9),
        5: dict(t_value=2),
        3: dict(t_value=1),
        # For W=1, static redistribution is binary in practice: 0=off, 1=on.
        1: dict(t_value=1),
    }

    def __init__(
        self,
        *,
        group_types: List[str] = None,
        population_biases: List[float] = None,
        ndcg_ks: List[int] = None,
        static_params: Dict[int, dict] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.population_biases = (
            population_biases if population_biases is not None else list(type(self).DEFAULT_POPULATION_BIASES)
        )
        self.ndcg_ks = ndcg_ks if ndcg_ks is not None else list(type(self).DEFAULT_NDCG_KS)
        self.static_params = static_params if static_params is not None else dict(type(self).DEFAULT_STATIC_PARAMS)

    def _t_value(self) -> int:
        if self.w_size not in self.static_params:
            raise ValueError(f"No STATIC_PARAMS for W_SIZE={self.w_size}")
        return self.static_params[self.w_size]["t_value"]

    def compute_results(self) -> Dict[str, Dict[str, Any]]:
        t_value = self._t_value()
        results: Dict[str, Dict[str, Any]] = {}
        ctx = get_context("spawn")
        n_gt = len(self.group_types)
        nb = len(self.population_biases)
        pool_args = [
            (
                i,
                n_gt,
                nb,
                gt,
                self.eval_type,
                self.groups_count,
                self.w_size,
                t_value,
                self.ndcg_ks,
                self.population_biases,
            )
            for i, gt in enumerate(self.group_types, start=1)
        ]
        with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
            for gt, res in pool.imap_unordered(_compute_one_group, pool_args, chunksize=1):
                results[gt] = res
        return results

    def make_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        t_value = self._t_value()
        cap = (
            rf"RFC podle typu skupiny a population bias; async static policy, group EASER, "
            rf"$t={t_value}$, $w={self.w_size}$."
        )
        return latex_rfc_table_group_types_by_biases(
            results=results,
            group_types=self.group_types,
            metric_key="average",
            caption=cap,
            label="tab:async_static_group_rfc_by_bias",
        )


if __name__ == "__main__":
    autorun(EvalAsyncStaticPolicySimplePriorityGroupRec)
