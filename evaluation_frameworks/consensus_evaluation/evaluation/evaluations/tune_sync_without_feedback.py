"""
**Tune — sync mediator without feedback (group same-EASer + aggregated lists).**

Explores discrete knobs on the sync-without-feedback stack; validation split defaults for speed.
"""

from typing import Any, Dict, List, Literal

import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    rfc_average_from_tune_result_cell,
    tune_group_types_present,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineGroupAllSameEaser
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorSyncApproach
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    SyncMediatorFactoryBuilderSync,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
    GR_AggregatedRecommendations,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


class TuneSyncWithoutFeedback(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 5
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_AGG_STRATEGIES: List[str] = ["mean", "min", "max", "median", "plurality"]
    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]

    def __init__(
        self,
        *,
        agg_strategies: List[str] = None,
        group_types: List[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.agg_strategies = (
            agg_strategies if agg_strategies is not None else list(type(self).DEFAULT_AGG_STRATEGIES)
        )
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)

    def compute_results(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {gt: {} for gt in self.group_types}
        r = Runner()
        for group_type in self.group_types:
            r.refresh_context(self.eval_type, group_type)
            for agg_strategy in self.agg_strategies:
                factory_method = lambda single_user_model, evaluation_set_csr, strat=agg_strategy: (
                    SyncMediatorFactoryBuilderSync()
                    .with_group_algorithm(lambda: GR_AggregatedRecommendations(single_user_model))
                    .with_group_recommender_engine(
                        lambda group, eg: RecommendationEngineGroupAllSameEaser(
                            group, evaluation_set_csr, eg, strat
                        )
                    )
                    .with_mediator(
                        lambda group, gre: ConsensusMediatorSyncApproach(group, self.w_size, gre)
                    )
                    .build()
                )
                res = r.run(factory_method, self.groups_count, window_size=self.w_size)
                results[group_type][agg_strategy] = res
        return results

    def make_table(self, results: dict) -> str:
        group_cols = tune_group_types_present(results, self.group_types)
        if not group_cols:
            return (
                "% tab:strategy_comparison: v uložených výsledcích chybí očekávané group_type klíče "
                "(prázdná tabulka). Zkus MODE=compute nebo sladit GROUPS_COUNT/W s původním tune během.\n"
            )
        data: Dict[str, Any] = {"Strategy": self.agg_strategies}
        for gt in group_cols:
            col = []
            gt_block = results.get(gt, {}) or {}
            for strategy in self.agg_strategies:
                col.append(
                    rfc_average_from_tune_result_cell(
                        gt_block.get(strategy) if isinstance(gt_block, dict) else None
                    )
                )
            data[gt] = col
        cols = ["Strategy"] + group_cols
        df = pd.DataFrame(data)[cols]
        df["average"] = df[group_cols].mean(axis=1, skipna=True)
        generator = LaTeXTableGeneratorSIUnitx(
            df,
            column_specs=None,
            column_width=1.5,
        )
        return generator.generate_table(
            caption=r"Srovnání strategií synchronního doporučovače skrze metriky RFC.",
            label="tab:strategy_comparison",
            cell_bold_fn=lambda row_idx, col_idx, val: (
                col_idx >= 1
                and pd.notna(val)
                and val == df.iloc[:, col_idx].min(skipna=True)
            ),
        )


if __name__ == "__main__":
    autorun(TuneSyncWithoutFeedback)
