from typing import Any, Dict, List
import argparse
import math
import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood import (
    order_algo_modules_paper,
    short_name_from_algo,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx

# ----- CONFIG -----

DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]
EVAL_TYPE: str = "test"

ALGOS: List[str] = [
    "eval_hybrid_general_rec_individual.py",
    "eval_hybrid_updatable.py",
    "eval_async_static_policy_simple_priority_function_group_rec.py",
    "eval_async_static_policy_simple_priority_function_individual_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
    "eval_sync_without_feedback.py",
    "eval_sync_with_feedback_ema.py",
]


def strategy_2_slug(algos: List[str]) -> Dict[str, str]:
    res = {}
    sync_ctr = 0
    async_ctr = 0
    hybrid_ctr = 0
    for algo_name in algos:
        if "async" in algo_name:
            res[f"A{async_ctr}"] = algo_name
            async_ctr += 1
        elif "hybrid" in algo_name:
            res[f"H{hybrid_ctr}"] = algo_name
            hybrid_ctr += 1
        else:
            res[f"S{sync_ctr}"] = algo_name
            sync_ctr += 1
    return res


def _safe_unmatched_ratio(stats: Dict[str, Any]) -> float:
    """
    Prefer explicit unmatched ratio if present; otherwise derive from
    total_rounds and matched_groups length.
    """
    if not isinstance(stats, dict):
        return math.nan

    if "unmatched_ratio" in stats:
        try:
            return float(stats["unmatched_ratio"])
        except (TypeError, ValueError):
            pass

    total = stats.get("total_rounds")
    matched = stats.get("matched_groups")
    if isinstance(total, (int, float)) and total > 0:
        if isinstance(matched, (list, tuple)):
            return max(0.0, 1.0 - (len(matched) / float(total)))
        # In this project, matched_groups is often a numpy.ndarray of rounds.
        if hasattr(matched, "__len__") and not isinstance(matched, (str, bytes, dict)):
            try:
                return max(0.0, 1.0 - (len(matched) / float(total)))
            except TypeError:
                pass
        if isinstance(matched, (int, float)):
            return max(0.0, 1.0 - (float(matched) / float(total)))

    return math.nan


def _pick_bias_block(group_data: Dict[Any, Any], bias: float) -> Dict[str, Any]:
    if not isinstance(group_data, dict):
        return {}
    if bias in group_data:
        return group_data[bias]
    bias_str = str(bias)
    if bias_str in group_data:
        return group_data[bias_str]
    # fallback: first available bias (keeps script useful for old caches)
    for _, val in group_data.items():
        if isinstance(val, dict):
            return val
    return {}


def load_unmatched_values(
    algos: List[str],
    window_size: str,
    eval_type: str = EVAL_TYPE,
    bias: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for algo_name in algos:
        try:
            loaded = load_eval_res(algo_name, window_size, eval_type)
        except Exception as e:
            print(f"WARNING: Evaluation results for {algo_name} not found ({e}).")
            continue

        out[algo_name] = {}
        present_group_types = [
            k for k in (loaded.keys() if isinstance(loaded, dict) else []) if isinstance(k, str)
        ] or list(DEFAULT_GROUP_TYPES)
        for gt in present_group_types:
            group_data = loaded.get(gt, {})
            bias_block = _pick_bias_block(group_data, bias)
            out[algo_name][gt] = _safe_unmatched_ratio(bias_block)

    return out


def create_table(
    values: Dict[str, Dict[str, float]],
    algos: List[str],
    group_types: List[str],
    *,
    as_percent: bool = True,
) -> str:
    slug2algo = strategy_2_slug(algos)
    rows = []
    for algo_name in order_algo_modules_paper(list(slug2algo.values())):
        row: Dict[str, Any] = {"algorithm": short_name_from_algo(algo_name)}
        for gt in group_types:
            val = values.get(algo_name, {}).get(gt, math.nan)
            if as_percent and pd.notna(val):
                val = val * 100.0
            row[gt] = val
        row["average"] = pd.Series([row[g] for g in group_types], dtype=float).mean(skipna=True)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["algorithm"] + group_types + ["average"])

    generator = LaTeXTableGeneratorSIUnitx(
        df,
        column_specs=None,
        column_width=1.5,
    )

    unit = "\\%" if as_percent else "ratio"
    return generator.generate_table(
        caption=rf"Podíl skupin bez shody do max. počtu kol ({unit}; nižší je lepší).",
        label="tab:unmatched_groups_comparison",
        cell_bold_fn=lambda row_idx, col_idx, val: (
            col_idx >= 1 and pd.notna(val) and val == df.iloc[:, col_idx].min(skipna=True)
        ),
    )


def create_slug_table(algos: List[str]) -> str:
    slug2algo = strategy_2_slug(algos)
    algo_to_slug = {algo: slug for slug, algo in slug2algo.items()}
    df = pd.DataFrame(
        [
            {"slug": algo_to_slug[algo], "algorithm": algo}
            for algo in order_algo_modules_paper(list(slug2algo.values()))
        ],
        columns=["slug", "algorithm"],
    )
    generator = LaTeXTableGeneratorSIUnitx(df, column_specs=None, column_width=3.0)
    return generator.generate_table(
        caption=r"Mapování zkratek (slugs) na algoritmy.",
        label="tab:slug_mapping_unmatched",
    )


def detect_group_types(values: Dict[str, Dict[str, float]], fallback: List[str]) -> List[str]:
    detected = set()
    for _, algo_vals in values.items():
        if not isinstance(algo_vals, dict):
            continue
        for k in algo_vals.keys():
            if isinstance(k, str):
                detected.add(k)
    if not detected:
        return list(fallback)
    preferred = [g for g in fallback if g in detected]
    rest = sorted(g for g in detected if g not in preferred)
    return preferred + rest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", default=10, help="Window size")
    parser.add_argument("--bias", type=float, default=0.0, help="Population bias key to read (default 0.0)")
    parser.add_argument(
        "--as-ratio",
        action="store_true",
        default=False,
        help="Print unmatched values as ratio in [0,1] instead of percent.",
    )
    args = parser.parse_args()

    print(
        f"Printing unmatched-groups table for window size {args.window_size}, "
        f"eval type {EVAL_TYPE}, bias {args.bias}"
    )
    values = load_unmatched_values(ALGOS, str(args.window_size), eval_type=EVAL_TYPE, bias=args.bias)
    group_types = detect_group_types(values, DEFAULT_GROUP_TYPES)

    print(create_slug_table(ALGOS))
    print()
    print(create_table(values, ALGOS, group_types=group_types, as_percent=not args.as_ratio))
