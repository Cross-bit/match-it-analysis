import gc
from multiprocessing import get_context
from typing import Any, Dict, List, Literal

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    latex_rfc_table_group_types_by_biases,
    pick_bias_result_for_table,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender import RecommendationEngineGroupAllSameEaserWithFeedback
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorSyncApproach
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    SyncMediatorFactoryBuilderSync,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_data_interpreter import (
    print_evaluation_result,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.batch_run_progress import (
    set_progress_slot_before_runner,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import RunnerLargeGroups
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedProfilesUpdatable,
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
        group_size,
        rec_window_size,
        agg_strategy,
        ema_alpha,
        ndcg_ks,
        population_biases,
    ) = args

    set_progress_slot_before_runner(slot_idx, n_slots, n_bias)
    r = RunnerLargeGroups()
    r.refresh_context(eval_type, group_size)

    factory_method = lambda single_user_model, evaluation_set_csr, alpha_=ema_alpha: (
        SyncMediatorFactoryBuilderSync()
        .with_group_algorithm(
            lambda: GR_AggregatedProfilesUpdatable(
                single_user_model, update_mode="ema", alpha=alpha_
            )
        )
        .with_group_recommender_engine(
            lambda group, eg: RecommendationEngineGroupAllSameEaserWithFeedback(group, eg, agg_strategy)
        )
        .with_mediator(lambda group, gre: ConsensusMediatorSyncApproach(group, rec_window_size, gre))
        .build()
    )

    res = r.run(
        factory_method,
        groups_count,
        ndcg_ks,
        population_biases,
        window_size=rec_window_size,
    )

    del r
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

    return group_type, res


class EvalLargeSyncWithFeedbackEma(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "test"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100
    DEFAULT_GROUP_SIZE = 10

    DEFAULT_AGG_STRATEGY = "mean"
    DEFAULT_EMA_ALPHA = 0.3  # jako eval_sync_with_feedback_ema (core)
    DEFAULT_GROUP_TYPES: List[str] = ["random"]
    DEFAULT_POPULATION_BIASES = [0]
    DEFAULT_NDCG_KS = [5, 10, 20, 50, 100]

    def __init__(
        self,
        *,
        agg_strategy: str = None,
        ema_alpha: float = None,
        group_types: List[str] = None,
        population_biases: List[float] = None,
        ndcg_ks: List[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.agg_strategy = agg_strategy if agg_strategy is not None else type(self).DEFAULT_AGG_STRATEGY
        self.ema_alpha = ema_alpha if ema_alpha is not None else type(self).DEFAULT_EMA_ALPHA
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.population_biases = (
            population_biases if population_biases is not None else list(type(self).DEFAULT_POPULATION_BIASES)
        )
        self.ndcg_ks = ndcg_ks if ndcg_ks is not None else list(type(self).DEFAULT_NDCG_KS)

    def compute_results(self) -> Dict[str, Dict[str, Any]]:
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
                self.group_size,
                self.w_size,
                self.agg_strategy,
                self.ema_alpha,
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
        for gt in self.group_types:
            print_evaluation_result(pick_bias_result_for_table(results[gt]))
        cap = (
            rf"RFC podle typu skupiny a population bias; velké skupiny ($n={self.group_size}$), "
            rf"sync s EMA zpětnou vazbou, agg={self.agg_strategy!r}, $\\alpha={self.ema_alpha}$, $w={self.w_size}$."
        )
        return latex_rfc_table_group_types_by_biases(
            results=results,
            group_types=self.group_types,
            metric_key="average",
            caption=cap,
            label="tab:large_sync_ema_rfc_by_bias",
        )


if __name__ == "__main__":
    autorun(EvalLargeSyncWithFeedbackEma)
