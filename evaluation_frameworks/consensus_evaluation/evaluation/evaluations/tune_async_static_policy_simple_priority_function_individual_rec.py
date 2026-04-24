"""
**Tune — async + static threshold + individual EASer.**

Sweeps static ``t_value`` (and related) grids; output is ranking-style tables for picking defaults used
in ``eval_async_static_policy_simple_priority_function_individual_rec``.
"""

from typing import Any, Dict, List, Literal

import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    rfc_average_from_tune_result_cell,
    tune_group_types_present,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ThresholdPolicyStatic
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    AsyncMediatorFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


class TuneAsyncStaticPolicySimplePriorityIndividualRec(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]

    def __init__(self, *, group_types: List[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)

    def _t_values_static_threshold(self) -> List[int]:
        if self.w_size <= 1:
            return [0, 1] if self.w_size == 1 else [0]
        return list(range(1, self.w_size + 1))

    def compute_results(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        t_values = self._t_values_static_threshold()
        results: Dict[str, Dict[int, Dict[str, Any]]] = {gt: {} for gt in self.group_types}
        for group_type in self.group_types:
            r = Runner()
            r.refresh_context(self.eval_type, group_type)
            for t in t_values:
                factory_method = lambda single_user_model, evaluation_set_csr, t_=t, w=self.w_size: (
                    AsyncMediatorFactoryBuilder()
                    .with_recommender_engine(
                        lambda group: RecommendationEngineIndividualEaser(group, model_iterator=single_user_model)
                    )
                    .with_priority_function(
                        lambda group, model=single_user_model: SimplePriorityFunction(group, algorithm=model)
                    )
                    .with_threshold_policy(lambda tv=t_: ThresholdPolicyStatic(t_param=tv))
                    .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
                    .with_mediator(
                        lambda group, rec, ru, th, w_=w: ConsensusMediatorAsyncApproach(
                            group, rec, ru, th, window_size=w_
                        )
                    )
                    .build()
                )
                res = r.run(factory_method, self.groups_count, window_size=self.w_size)
                results[group_type][t] = res
        return results

    def make_table(self, results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
        t_values = self._t_values_static_threshold()
        group_cols = tune_group_types_present(results, self.group_types)
        if not group_cols:
            return (
                "% tab:async_static_threshold_grid: v uložených výsledcích chybí očekávané group_type klíče "
                "(prázdná tabulka). Zkus MODE=compute nebo sladit GROUPS_COUNT/W s původním tune během.\n"
            )
        data: Dict[str, Any] = {"t": t_values}
        for gt in group_cols:
            col = []
            gt_block = results.get(gt, {}) or {}
            for t in t_values:
                col.append(rfc_average_from_tune_result_cell(gt_block.get(t) if isinstance(gt_block, dict) else None))
            data[gt] = col
        cols = ["t"] + group_cols
        df = pd.DataFrame(data)[cols]
        df["average"] = df[group_cols].mean(axis=1, skipna=True)
        generator = LaTeXTableGeneratorSIUnitx(
            df,
            column_specs=None,
            column_width=1.5,
        )
        return generator.generate_table(
            caption=rf"Vyhodnocení metrik RFC pro asynchronní doporučovač se statickým parametrem $t$ pro velikost okna $w={self.w_size}$ a individuálním doporučovačem pro každého uživatele.",
            label="tab:async_static_threshold_grid",
            cell_bold_fn=lambda row_idx, col_idx, val: (
                col_idx >= 1
                and pd.notna(val)
                and val == df.iloc[:, col_idx].min(skipna=True)
            ),
        )


if __name__ == "__main__":
    autorun(TuneAsyncStaticPolicySimplePriorityIndividualRec)
