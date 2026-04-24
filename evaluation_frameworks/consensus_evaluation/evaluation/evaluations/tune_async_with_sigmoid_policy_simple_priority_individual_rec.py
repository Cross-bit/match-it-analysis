"""
**Tune / validation — async + sigmoid threshold + individual EASer.**

Grid-searches sigmoid hyperparameters (and related knobs) per ``W`` and group type; aggregates mean
rounds-to-consensus into DataFrames / LaTeX via ``LaTeXTableGeneratorSIUnitx``. Not the default
``eval-full`` orchestration — use for choosing ``DEFAULT_SIGMOID_PARAMS`` in the matching ``eval_*``.
"""

from itertools import product
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import ConsensusExperimentBase
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ThresholdPolicySigmoid
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import (
    AsyncMediatorFactoryBuilder,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


def _results_to_df(results: Dict[str, Dict[tuple, Dict[str, Any]]], group_types: List[str]) -> pd.DataFrame:
    rows = []
    for gt in group_types:
        if gt not in results:
            continue
        for key, metrics in results[gt].items():
            rows.append({"params": key, gt: metrics[0]["average"]})
    df = pd.DataFrame(rows).groupby("params", as_index=False).first()
    cols = ["params"] + [gt for gt in group_types if gt in df.columns]
    df = df[cols]
    if any(c in df.columns for c in group_types):
        df["average"] = df[group_types].mean(axis=1, skipna=True)
    return df


def summarize_results(
    results: Dict[str, Dict[tuple, Dict[str, Any]]],
    group_types: List[str],
    top_k: int = 10,
) -> Tuple[str, Dict[str, Any]]:
    df = _results_to_df(results, group_types)
    if "average" not in df.columns or df.empty:
        return "⚠️ Nemám co sumarizovat (prázdná data).", {}

    overall_mean = float(df["average"].mean(skipna=True))
    overall_var = float(df["average"].var(ddof=1)) if len(df) > 1 else 0.0
    overall_std = float(np.sqrt(overall_var))

    min_idx = int(df["average"].idxmin(skipna=True))
    max_idx = int(df["average"].idxmax(skipna=True))
    best_row = df.loc[min_idx]
    worst_row = df.loc[max_idx]

    top_k = max(1, min(top_k, len(df)))
    top_best_df = df.nsmallest(top_k, "average")
    top_worst_df = df.nlargest(top_k, "average")

    per_group = {}
    for gt in group_types:
        if gt in df.columns:
            series = df[gt].dropna()
            if not series.empty:
                gmin_idx = int(series.idxmin())
                gmax_idx = int(series.idxmax())
                per_group[gt] = {
                    "min": {"params": tuple(df.loc[gmin_idx, "params"]), "value": float(df.loc[gmin_idx, gt])},
                    "max": {"params": tuple(df.loc[gmax_idx, "params"]), "value": float(df.loc[gmax_idx, gt])},
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                }

    def _fmt_param(p):
        return f"{tuple(p)}"

    def _fmt_row(row: pd.Series) -> str:
        return f"{_fmt_param(row['params'])} → average={row['average']:.6f}"

    best_text_lines = [f"{i+1}. {_fmt_row(r)}" for i, (_, r) in enumerate(top_best_df.iterrows())]
    worst_text_lines = [f"{i+1}. {_fmt_row(r)}" for i, (_, r) in enumerate(top_worst_df.iterrows())]

    text = []
    text.append("📊 Základní statistiky (napříč všemi řádky):")
    text.append(f"• Skutečný průměr všech průměrů: {overall_mean:.6f}")
    text.append(f"• Rozptyl 'average': {overall_var:.6f}  (σ={overall_std:.6f})")
    text.append(f"• Globální minimum: {_fmt_row(best_row)}")
    text.append(f"• Globální maximum: {_fmt_row(worst_row)}")
    text.append("")
    text.append(f"🥇 TOP {top_k} nejlepších (lowest average):")
    text.extend(best_text_lines)
    text.append("")
    text.append(f"🥵 TOP {top_k} nejhorších (highest average):")
    text.extend(worst_text_lines)
    text.append("")
    text.append("🔎 Per-sloupec:")
    for gt, st in per_group.items():
        text.append(
            f"• {gt}: min={st['min']['value']:.6f} @ {st['min']['params']}, "
            f"max={st['max']['value']:.6f} @ {st['max']['params']}, "
            f"mean={st['mean']:.6f}, σ={st['std']:.6f}"
        )

    out_dict = {
        "overall": {
            "mean_of_averages": overall_mean,
            "variance_of_averages": overall_var,
            "std_of_averages": overall_std,
            "global_min": {"params": tuple(best_row["params"]), "average": float(best_row["average"])},
            "global_max": {"params": tuple(worst_row["params"]), "average": float(worst_row["average"])},
        },
        "top_best": [
            {"params": tuple(r["params"]), "average": float(r["average"])} for _, r in top_best_df.iterrows()
        ],
        "top_worst": [
            {"params": tuple(r["params"]), "average": float(r["average"])} for _, r in top_worst_df.iterrows()
        ],
        "per_group": per_group,
    }

    return "\n".join(text), out_dict


class TuneAsyncWithSigmoidPolicySimplePriorityIndividualRec(ConsensusExperimentBase):
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_W_SIZE = 10
    DEFAULT_GROUPS_COUNT = 100

    DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
    DEFAULT_SIGMOID_PARAMS = {
        10: dict(
            center=[1, 2, 5],
            steepness=[0.2, 0.75, 1.4],
            c_init=[0, 0.2, 0.4, 0.7],
            max_fill=[1, 2, 5, 10, 15, 20],
            min_fill=[0],
        ),
        5: dict(
            center=[1, 2],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2, 0.4],
            max_fill=[1, 2, 5, 10],
            min_fill=[0],
        ),
        3: dict(
            center=[1, 2],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2, 0.4],
            max_fill=[1, 2, 3],
            min_fill=[0],
        ),
        1: dict(
            center=[1],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2, 0.4],
            max_fill=[1],
            min_fill=[0],
        ),
    }
    # --fast-grid: ~4.5× méně kombinací než DEFAULT (W=10: 48 vs 216 na group_type)
    FAST_SIGMOID_PARAMS = {
        10: dict(
            center=[1, 5],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2, 0.7],
            max_fill=[1, 5, 10, 20],
            min_fill=[0],
        ),
        5: dict(
            center=[1, 2],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2],
            max_fill=[1, 5, 10],
            min_fill=[0],
        ),
        3: dict(
            center=[1, 2],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2],
            max_fill=[1, 3],
            min_fill=[0],
        ),
        1: dict(
            center=[1],
            steepness=[0.2, 0.75],
            c_init=[0, 0.2],
            max_fill=[1],
            min_fill=[0],
        ),
    }

    def __init__(
        self,
        *,
        group_types: List[str] = None,
        sigmoid_params: Dict[int, dict] = None,
        fast_grid: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.group_types = group_types if group_types is not None else list(type(self).DEFAULT_GROUP_TYPES)
        if sigmoid_params is not None:
            self.sigmoid_params = sigmoid_params
        elif fast_grid:
            self.sigmoid_params = {k: dict(v) for k, v in type(self).FAST_SIGMOID_PARAMS.items()}
        else:
            self.sigmoid_params = {k: dict(v) for k, v in type(self).DEFAULT_SIGMOID_PARAMS.items()}

    def _param_lists(self) -> Dict[str, List[Any]]:
        if self.w_size not in self.sigmoid_params:
            raise ValueError(f"No SIGMOID_PARAMS for W_SIZE={self.w_size}")
        return self.sigmoid_params[self.w_size]

    def compute_results(self) -> Dict[str, Dict[tuple, Dict[str, Any]]]:
        p = self._param_lists()
        results: Dict[str, Dict[tuple, Dict[str, Any]]] = {gt: {} for gt in self.group_types}

        for group_type in self.group_types:
            r = Runner()
            r.refresh_context(self.eval_type, group_type)

            for center, steepness, c_init, max_fill, min_fill in product(
                p["center"], p["steepness"], p["c_init"], p["max_fill"], p["min_fill"]
            ):
                factory_method = lambda single_user_model, evaluation_set_csr, center_=center, steepness_=steepness, c_init_=c_init, max_fill_=max_fill, min_fill_=min_fill, w=self.w_size: (
                    AsyncMediatorFactoryBuilder()
                    .with_recommender_engine(
                        lambda group: RecommendationEngineIndividualEaser(
                            group, model_iterator=single_user_model
                        )
                    )
                    .with_priority_function(
                        lambda group, model=single_user_model: SimplePriorityFunction(
                            group, algorithm=model
                        )
                    )
                    .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
                    .with_sigmoid_policy(
                        lambda ru,
                        center__=center_,
                        steepness__=steepness_,
                        c_init__=c_init_,
                        max_fill__=max_fill_,
                        min_fill__=min_fill_: ThresholdPolicySigmoid(
                            red_context=getattr(ru, "redistribution_context", ru),
                            window_size=w,
                            sigmoid_center=center__,
                            sigmoid_steepness=steepness__,
                            c_init=c_init__,
                            max_filling=max_fill__,
                            min_filling=min_fill__,
                        )
                    )
                    .with_mediator(
                        lambda group, rec, ru, th, w_=w: ConsensusMediatorAsyncApproach(
                            group, rec, ru, th, window_size=w_
                        )
                    )
                    .build()
                )

                res = r.run(factory_method, self.groups_count, window_size=self.w_size)
                results[group_type][(center, steepness, c_init, max_fill, min_fill)] = res

        return results

    def stats(self, results) -> str:
        report, _ = summarize_results(results, self.group_types, top_k=10)
        return report

    def make_table(self, results: Dict[str, Dict[tuple, Dict[str, Any]]]) -> str:
        rows = []
        for gt in self.group_types:
            for key, metrics in results[gt].items():
                avg = metrics[0]["average"]
                rows.append({"params": key, gt: avg})
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
    autorun(TuneAsyncWithSigmoidPolicySimplePriorityIndividualRec)
