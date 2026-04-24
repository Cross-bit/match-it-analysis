"""
**Tune — alternate mean-feedback experiment (underscore module).**

Second exploration track for mean-based profile updates (``TuneSyncWithFeedbackMeanUnderscore``);
kept separate from ``tune_sync_with_feedback_mean.py`` so Makefile / history can target either.
"""

from typing import Any, Dict, List, Literal

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


class TuneSyncWithFeedbackMeanUnderscore(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_AGG_STRATEGIES: List[str] = ["mean", "min", "max", "median", "plurality"]
    DEFAULT_GROUP_TYPES: List[str] = ["similar"]
    DEFAULT_NDCG_KS = [20]

    def __init__(
        self,
        *,
        agg_strategies: List[str] = None,
        group_types: List[str] = None,
        ndcg_ks: List[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.agg_strategies = (
            agg_strategies if agg_strategies is not None else list(type(self).DEFAULT_AGG_STRATEGIES)
        )
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        self.ndcg_ks = ndcg_ks if ndcg_ks is not None else list(type(self).DEFAULT_NDCG_KS)

    def compute_results(self) -> dict:
        results = {gt: {} for gt in self.group_types}
        for group_type in self.group_types:
            r = Runner()
            r.refresh_context(self.eval_type, group_type)
            for agg_strategy in self.agg_strategies:
                factory_method = lambda single_user_model, evaluation_set_csr, strat=agg_strategy: (
                    SyncMediatorFactoryBuilderSync()
                    .with_group_algorithm(
                        lambda: GR_AggregatedProfilesUpdatable(single_user_model, update_mode="mean")
                    )
                    .with_group_recommender_engine(
                        lambda group, eg: RecommendationEngineGroupAllSameEaserWithFeedback(group, eg, strat)
                    )
                    .with_mediator(
                        lambda group, gre: ConsensusMediatorSyncApproach(group, self.w_size, gre)
                    )
                    .build()
                )
                res = r.run(
                    factory_method,
                    self.groups_count,
                    self.ndcg_ks,
                    window_size=self.w_size,
                )
                results[group_type][agg_strategy] = res
        return results

    def make_table(self, results: dict) -> str:
        k = None
        for gt in self.group_types:
            for strategy in self.agg_strategies:
                nd = results.get(gt, {}).get(strategy, {}).get("ndcg")
                if nd and "k" in nd:
                    k = nd["k"]
                    break
            if k is not None:
                break
        k = k if k is not None else "?"

        rows = []
        for strategy in self.agg_strategies:
            row = {"Strategy": strategy}
            for gt in self.group_types:
                nd = results[gt][strategy].get("ndcg", {})
                row[f"{gt}-mean"] = nd.get("per_user_ndcg_mean_overall", 0.0)
                row[f"{gt}-min"] = nd.get("per_user_ndcg_min_overall", 0.0)
                row[f"{gt}-max"] = nd.get("per_user_ndcg_max_overall", 0.0)
            rows.append(row)
        df_indiv = pd.DataFrame(rows)

        gen_indiv = LaTeXTableGeneratorSIUnitx(
            df_indiv, column_specs=[(1, 4)] * len(df_indiv.columns), column_width=1.5
        )
        latex_indiv = gen_indiv.generate_table(
            caption=rf"Individual NDCG@{k} averaged over groups (mean/min/max).",
            label="tab:ndcg_individual",
        )

        rows = []
        for strategy in self.agg_strategies:
            row = {"Strategy": strategy}
            for gt in self.group_types:
                nd = results[gt][strategy].get("ndcg", {})
                row[f"{gt}-mean"] = nd.get("ndcg_com_mean_overall", 0.0)
                row[f"{gt}-min"] = nd.get("ndcg_com_min_overall", 0.0)
            rows.append(row)
        df_com = pd.DataFrame(rows)

        gen_com = LaTeXTableGeneratorSIUnitx(
            df_com, column_specs=[(1, 4)] * len(df_com.columns), column_width=1.5
        )
        latex_com = gen_com.generate_table(
            caption=rf"Common-GT NDCG@{k} averaged over groups (mean/min).",
            label="tab:ndcg_common",
        )

        return "\n\n".join([latex_indiv, latex_com])


if __name__ == "__main__":
    autorun(TuneSyncWithFeedbackMeanUnderscore)
