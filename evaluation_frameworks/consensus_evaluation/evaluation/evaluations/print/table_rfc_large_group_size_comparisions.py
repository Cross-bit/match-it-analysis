from typing import Any, Dict, List, Optional, Sequence
import argparse
import math
import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import (
    evaluation_results_dir,
    load_eval_res,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.rfc_table_metric_spec import (
    add_rfc_metric_arg,
    resolve_rfc_metric,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


EVAL_TYPE: str = "test"
BIAS_KEY = 0.0

ALGOS: List[str] = [
    "eval_large_hybrid_general_rec_individual.py",
    "eval_large_hybrid_group_updatable.py",
    "eval_large_sync_without_feedback.py",
    "eval_large_sync_with_feedback_ema.py",
    # Canonical async set for main large-group table: A0,A1,A2.
    "eval_async_static_policy_simple_priority_function_group_rec.py",
    "eval_async_static_policy_simple_priority_function_individual_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
    # NOTE: eval_async_with_sigmoid_policy_simple_priority_group_rec.py is AX (extra),
    # intentionally excluded from the main canonical table.
]

SHORT_NAME_BY_ALGO: Dict[str, str] = {
    "eval_async_static_policy_simple_priority_function_individual_rec.py": "Async-Static-Ind",
    "eval_async_static_policy_simple_priority_function_group_rec.py": "Async-Static-Grp",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py": "Async-Dyn-Ind",
    "eval_large_hybrid_general_rec_individual.py": "Hybrid-Ind",
    "eval_large_hybrid_group_updatable.py": "Hybrid-Grp-EMA",
    "eval_large_sync_without_feedback.py": "Sync",
    "eval_large_sync_with_feedback_ema.py": "Sync-EMA",
}


def strategy_2_slug(algos: List[str]) -> Dict[str, str]:
    # Stable slug mapping aligned with canonical naming:
    # A0=async-static-group, A1=async-static-individual, A2=async-sigmoid-individual,
    # S0=sync-no-feedback, S1=sync-ema, H0/H1=hybrid variants.
    fixed = {
        "eval_large_hybrid_general_rec_individual.py": "H0",
        "eval_large_hybrid_group_updatable.py": "H1",
        "eval_large_sync_without_feedback.py": "S0",
        "eval_large_sync_with_feedback_ema.py": "S1",
        "eval_async_static_policy_simple_priority_function_group_rec.py": "A0",
        "eval_async_static_policy_simple_priority_function_individual_rec.py": "A1",
        "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py": "A2",
        # extra (not in main canonical table)
        "eval_async_with_sigmoid_policy_simple_priority_group_rec.py": "AX",
    }

    res: Dict[str, str] = {}
    used = set()
    sync_ctr = 2
    async_ctr = 3
    hybrid_ctr = 2
    for algo_name in algos:
        if algo_name in fixed:
            slug = fixed[algo_name]
            res[slug] = algo_name
            used.add(slug)
            continue
        if "async" in algo_name:
            while f"A{async_ctr}" in used:
                async_ctr += 1
            res[f"A{async_ctr}"] = algo_name
            used.add(f"A{async_ctr}")
            async_ctr += 1
        elif "hybrid" in algo_name:
            while f"H{hybrid_ctr}" in used:
                hybrid_ctr += 1
            res[f"H{hybrid_ctr}"] = algo_name
            used.add(f"H{hybrid_ctr}")
            hybrid_ctr += 1
        else:
            while f"S{sync_ctr}" in used:
                sync_ctr += 1
            res[f"S{sync_ctr}"] = algo_name
            used.add(f"S{sync_ctr}")
            sync_ctr += 1
    return res


def _safe_get_metric(d: Dict[str, Any], metric_key: str = "average") -> Optional[float]:
    if d is None:
        return math.nan
    if metric_key in d:
        try:
            return float(d[metric_key])
        except (TypeError, ValueError):
            pass
    metrics = d.get("metrics") if isinstance(d, dict) else None
    if isinstance(metrics, dict) and metric_key in metrics:
        try:
            return float(metrics[metric_key])
        except (TypeError, ValueError):
            pass
    return math.nan


def _short_name_from_algo(algo_name: str) -> str:
    return SHORT_NAME_BY_ALGO.get(algo_name, algo_name.replace(".py", ""))


def _pick_bias_block(group_data: Dict[Any, Any], bias: float) -> Dict[str, Any]:
    if not isinstance(group_data, dict):
        return {}
    if bias in group_data:
        return group_data[bias]
    bstr = str(bias)
    if bstr in group_data:
        return group_data[bstr]
    if 0 in group_data:
        return group_data[0]
    if "0" in group_data:
        return group_data["0"]
    for _, val in group_data.items():
        if isinstance(val, dict):
            return val
    return {}


def load_large_rfc_values(
    algos: List[str],
    *,
    window_size: str,
    eval_type: str,
    groups_count: int,
    group_sizes: List[int],
    metric_key: str = "average",
    verbose_misses: bool = False,
) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for algo_name in algos:
        out[algo_name] = {}
        for gs in group_sizes:
            try:
                loaded = load_eval_res(
                    algo_name,
                    window_size,
                    eval_type,
                    group_size=gs,
                    groups_count=groups_count,
                )
            except Exception as e:
                expected_labeled = evaluation_results_dir(
                    window_size=window_size,
                    eval_type=eval_type,
                    evaluation_name=algo_name,
                    group_size=gs,
                    groups_count=groups_count,
                    layout="labeled",
                )
                print(
                    f"WARNING: cache miss for {algo_name} @ group_size={gs}. "
                    f"Expected: {expected_labeled}"
                )
                if verbose_misses:
                    print(f"  detail: {e}")
                out[algo_name][gs] = math.nan
                continue

            group_block = loaded.get("random", {}) if isinstance(loaded, dict) else {}
            bias_block = _pick_bias_block(group_block, BIAS_KEY)
            out[algo_name][gs] = _safe_get_metric(bias_block, metric_key=metric_key)
    return out


def _filter_algos_with_any_values(values: Dict[str, Dict[int, float]], algos: List[str]) -> List[str]:
    kept: List[str] = []
    for a in algos:
        per_size = values.get(a, {})
        has_value = any(pd.notna(v) for v in per_size.values())
        if has_value:
            kept.append(a)
    return kept


def create_table(
    values: Dict[str, Dict[int, float]],
    algos: List[str],
    *,
    group_sizes: List[int],
    adjust_A: bool = True,
    apply_async_minus_one: bool = True,
    caption: Optional[str] = None,
) -> str:
    slug2algo = strategy_2_slug(algos)
    cols = [f"size_{gs}" for gs in group_sizes]
    rows = []
    for slug, algo_name in slug2algo.items():
        is_adj = apply_async_minus_one and adjust_A and ("A" in slug)
        label = _short_name_from_algo(algo_name)
        row: Dict[str, Any] = {"algorithm": label + (" (RFC$_{adj.}$)" if is_adj else "")}
        for gs, c in zip(group_sizes, cols):
            val = values.get(algo_name, {}).get(gs, math.nan)
            if is_adj and pd.notna(val):
                val = val - 1
            row[c] = val
        row["average"] = pd.Series([row[c] for c in cols], dtype=float).mean(skipna=True)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["algorithm"] + cols + ["average"])
    generator = LaTeXTableGeneratorSIUnitx(df, column_specs=None, column_width=1.5)
    cap = caption or rf"Porovnání large-group algoritmů skrze RFC pro různé velikosti skupin."
    return generator.generate_table(
        caption=cap,
        label="tab:large_group_size_rfc_comparison",
        cell_bold_fn=lambda row_idx, col_idx, val: (
            col_idx >= 1
            and pd.notna(val)
            and "adj." not in str(df.iloc[row_idx, 0])
            and val == df.loc[~df.iloc[:, 0].str.contains("adj.", regex=False), df.columns[col_idx]].min(skipna=True)
        ),
    )

def _avg_values_over_windows(
    *,
    windows: Sequence[int],
    algos: List[str],
    eval_type: str,
    groups_count: int,
    group_sizes: List[int],
    metric_key: str,
    verbose_misses: bool,
) -> Dict[str, Dict[int, float]]:
    """
    Aggregate RFC values over selected windows by simple mean (ignoring NaN).
    Output shape is compatible with create_table(): {algo: {group_size: value}}.
    """
    per_window: Dict[int, Dict[str, Dict[int, float]]] = {}
    for w in windows:
        per_window[w] = load_large_rfc_values(
            algos,
            window_size=str(w),
            eval_type=eval_type,
            groups_count=groups_count,
            group_sizes=group_sizes,
            metric_key=metric_key,
            verbose_misses=verbose_misses,
        )

    out: Dict[str, Dict[int, float]] = {a: {} for a in algos}
    for a in algos:
        for gs in group_sizes:
            vals = [per_window[w].get(a, {}).get(gs, math.nan) for w in windows]
            arr = pd.Series(vals, dtype=float)
            out[a][gs] = float(arr.mean(skipna=True)) if arr.notna().any() else math.nan
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=None, help="Single window W (legacy mode).")
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=None,
        help="Multiple windows (e.g. --windows 1 3 5 10).",
    )
    parser.add_argument("--groups-count", type=int, default=100, help="Evaluation groups count used in cache path")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[3, 5, 7, 10], help="Group sizes to compare")
    parser.add_argument(
        "--only-available",
        action="store_true",
        help="Show only algorithms that have at least one non-NaN value in the selected cache slice.",
    )
    parser.add_argument(
        "--verbose-misses",
        action="store_true",
        help="Print full exception details for cache misses.",
    )
    parser.add_argument(
        "--aggregate-windows",
        action="store_true",
        help="Aggregate table over selected --windows (mean across W for each algo and group size).",
    )
    add_rfc_metric_arg(parser)
    args = parser.parse_args()

    spec = resolve_rfc_metric(args.rfc_metric)
    if args.windows is not None and args.window_size is not None:
        raise ValueError("Use either --window-size or --windows, not both.")
    if args.windows is not None:
        windows = [int(w) for w in args.windows]
    elif args.window_size is not None:
        windows = [int(args.window_size)]
    else:
        windows = [10]

    if args.aggregate_windows:
        print(
            f"[table_rfc_large_group_size] AGG over windows={windows} metric={args.rfc_metric} "
            f"→ `{spec.storage_key}`"
        )
        vals = _avg_values_over_windows(
            windows=windows,
            algos=ALGOS,
            eval_type=EVAL_TYPE,
            groups_count=args.groups_count,
            group_sizes=list(args.group_sizes),
            metric_key=spec.storage_key,
            verbose_misses=args.verbose_misses,
        )
        table_algos = _filter_algos_with_any_values(vals, ALGOS) if args.only_available else ALGOS
        print(
            f"[table_rfc_large_group_size] included_algorithms={len(table_algos)}/{len(ALGOS)} "
            f"(only_available={args.only_available})"
        )
        print(
            create_table(
                vals,
                table_algos,
                group_sizes=list(args.group_sizes),
                apply_async_minus_one=spec.subtract_one_for_async_slug,
                caption=(
                    rf"Porovnání large-group algoritmů — {spec.latex_caption_cs} "
                    rf"(různé velikosti skupin; průměr přes okna $W \in \{{{','.join(str(w) for w in windows)}\}}$)."
                ),
            )
        )
    else:
        for w in windows:
            print(
                f"[table_rfc_large_group_size] W={w} metric={args.rfc_metric} "
                f"→ `{spec.storage_key}`"
            )
            vals = load_large_rfc_values(
                ALGOS,
                window_size=str(w),
                eval_type=EVAL_TYPE,
                groups_count=args.groups_count,
                group_sizes=list(args.group_sizes),
                metric_key=spec.storage_key,
                verbose_misses=args.verbose_misses,
            )
            table_algos = _filter_algos_with_any_values(vals, ALGOS) if args.only_available else ALGOS
            print(
                f"[table_rfc_large_group_size] included_algorithms={len(table_algos)}/{len(ALGOS)} "
                f"(only_available={args.only_available})"
            )
            print(
                create_table(
                    vals,
                    table_algos,
                    group_sizes=list(args.group_sizes),
                    apply_async_minus_one=spec.subtract_one_for_async_slug,
                    caption=(
                        rf"Porovnání large-group algoritmů — {spec.latex_caption_cs} "
                        rf"(různé velikosti skupin; $W={w}$)."
                    ),
                )
            )
