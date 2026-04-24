from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood import (
    ALGOS,
    EVAL_TYPE,
    GROUP_TYPES,
    display_name_for_slug,
    strategy_2_slug,
)


def _escape_tex(text: str) -> str:
    return text.replace("_", r"\_")


def _canonical_slug_order(slug2algo: Dict[str, str]) -> List[str]:
    preferred = ["H0", "H1", "A0", "A1", "A2", "S0", "S1"]
    present = [s for s in preferred if s in slug2algo]
    rest = [s for s in sorted(slug2algo.keys()) if s not in present]
    return present + rest


def _bias_key(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _bias_get(d: Dict[Any, Any], b: float) -> Any:
    for k, v in d.items():
        if _bias_key(k) == float(b):
            return v
    return None


def _extract_counts(stats: Dict[str, Any]) -> Tuple[float, float]:
    """
    Return (matched_count, total_rounds).
    matched_count is len(matched_groups) when available.
    """
    if not isinstance(stats, dict):
        return math.nan, math.nan
    total = stats.get("total_rounds", math.nan)
    matched_obj = stats.get("matched_groups", None)
    matched = math.nan
    if isinstance(matched_obj, (list, tuple, np.ndarray)):
        matched = float(len(matched_obj))
    elif isinstance(matched_obj, (int, float)):
        matched = float(matched_obj)
    try:
        total = float(total)
    except (TypeError, ValueError):
        total = math.nan
    return matched, total


def _cell(
    matched: float,
    total: float,
    rate_decimals: int = 2,
    percent_only: bool = False,
    ratio_only: bool = False,
) -> str:
    if not math.isfinite(matched) or not math.isfinite(total) or total <= 0:
        return "--"
    ratio = matched / total
    rate = 100.0 * ratio
    if ratio_only:
        return f"{ratio:.{rate_decimals}f}"
    if percent_only:
        return f"{rate:.{rate_decimals}f}\\%"
    return f"{int(round(matched))}/{int(round(total))} ({rate:.{rate_decimals}f}\\%)"


def _load_per_algo_group_bias(
    algo_name: str,
    w_size: int,
    eval_type: str,
    groups_count: int | None,
    group_types: List[str],
    merge_all_pickles: bool,
    group_size: int | None = None,
) -> Dict[str, Dict[float, Tuple[float, float]]]:
    """
    {group_type: {bias: (matched,total)}}
    """
    out: Dict[str, Dict[float, Tuple[float, float]]] = {gt: {} for gt in group_types}
    # Prefer explicit group_size slice when requested, but fall back to base cache
    # for modules/runs that do not store per-group-size paths.
    try:
        data = load_eval_res(
            algo_name,
            str(w_size),
            eval_type,
            group_size=group_size,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
        )
    except FileNotFoundError:
        if group_size is None:
            raise
        data = load_eval_res(
            algo_name,
            str(w_size),
            eval_type,
            group_size=None,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
        )
    if not isinstance(data, dict):
        return out
    for gt in group_types:
        gt_data = data.get(gt, {})
        if not isinstance(gt_data, dict):
            continue
        for b_key, stats in gt_data.items():
            b = _bias_key(b_key)
            if not isinstance(b, (int, float)):
                continue
            out[gt][float(b)] = _extract_counts(stats)
    return out


def _build_table(
    windows: List[int],
    biases: List[float],
    group_types: List[str],
    slug2algo: Dict[str, str],
    eval_type: str,
    groups_count: int | None,
    merge_all_pickles: bool,
    label: str,
    percent_only: bool,
    ratio_only: bool,
    decimals: int,
) -> str:
    block_cols = len(biases)
    col_spec = "l " + " ".join(["c" for _ in range(len(windows) * block_cols)])
    header_w_blocks = " & ".join(
        rf"\multicolumn{{{block_cols}}}{{c}}{{\textbf{{$W={w}$}}}}" for w in windows
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * block_cols}-{1 + (i + 1) * block_cols}}}"
        for i in range(len(windows))
    )
    header_biases_parts = []
    for _w in windows:
        header_biases_parts.extend([f"{b:.1f}" for b in biases])

    lines: List[str] = [
        r"\begin{table}",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r" & " + header_w_blocks + r" \\",
        cmidrules,
        r"\textbf{algorithm} & " + " & ".join(header_biases_parts) + r" \\",
        r"\midrule",
    ]

    ordered_slugs = _canonical_slug_order(slug2algo)
    for slug in ordered_slugs:
        algo = slug2algo[slug]
        row: List[str] = []
        for w in windows:
            per_gt = _load_per_algo_group_bias(
                algo,
                w,
                eval_type=eval_type,
                groups_count=groups_count,
                group_types=group_types,
                merge_all_pickles=merge_all_pickles,
            )
            for b in biases:
                matched_sum = 0.0
                total_sum = 0.0
                any_data = False
                for gt in group_types:
                    pair = _bias_get(per_gt.get(gt, {}), b)
                    if not pair:
                        continue
                    m, t = pair
                    if math.isfinite(m) and math.isfinite(t) and t > 0:
                        matched_sum += m
                        total_sum += t
                        any_data = True
                row.append(
                    _cell(
                        matched_sum,
                        total_sum,
                        rate_decimals=decimals,
                        percent_only=percent_only,
                        ratio_only=ratio_only,
                    )
                    if any_data
                    else "--"
                )
        label = display_name_for_slug(slug2algo, slug)
        lines.append(rf"{_escape_tex(label)} & " + " & ".join(row) + r" \\")

    groups_count_txt = str(groups_count) if groups_count is not None else "mixed"
    caption_prefix = (
        r"\caption{Success ratio shody (0--1) "
        if ratio_only
        else
        r"\caption{Success rate shody (\%) "
        if percent_only
        else r"\caption{Počet úspěšných shod (matched/total, v závorce success rate) "
    )
    caption_text = (
        caption_prefix
        + f"agregovaný přes typy skupin ({', '.join(group_types)}), split={eval_type}, "
        + f"groups\\_count={groups_count_txt}.}}"
    )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            caption_text,
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Print LaTeX table of successful matches (count + success rate)."
    )
    p.add_argument("--windows", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--biases", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    p.add_argument("--group-types", nargs="*", default=["similar", "outlier", "random"])
    p.add_argument("--eval-type", default=EVAL_TYPE, choices=["train", "validation", "test"])
    p.add_argument("--groups-count", type=int, default=None)
    p.add_argument("--latest-pickle-only", action="store_true")
    p.add_argument("--label", default="tab:success_matches_by_w_bias")
    p.add_argument("--decimals", type=int, default=2, help="Decimal places for rate/ratio values.")
    p.add_argument(
        "--percent-only",
        action="store_true",
        help="Print only success rate in percent (without matched/total counts).",
    )
    p.add_argument(
        "--ratio-only",
        action="store_true",
        help="Print only success ratio in [0,1] (without matched/total counts).",
    )
    args = p.parse_args()

    slug2algo = strategy_2_slug(ALGOS)
    print(
        _build_table(
            windows=list(args.windows),
            biases=list(args.biases),
            group_types=list(args.group_types) if args.group_types else list(GROUP_TYPES),
            slug2algo=slug2algo,
            eval_type=args.eval_type,
            groups_count=args.groups_count,
            merge_all_pickles=not args.latest_pickle_only,
            label=args.label,
            percent_only=args.percent_only,
            ratio_only=args.ratio_only,
            decimals=args.decimals,
        )
    )

