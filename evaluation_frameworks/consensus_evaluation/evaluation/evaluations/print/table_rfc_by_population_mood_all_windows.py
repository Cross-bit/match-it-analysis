from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Dict, List

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.rfc_table_metric_spec import (
    add_rfc_metric_arg,
    resolve_rfc_metric,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood import (
    ALGOS,
    EVAL_TYPE,
    GROUP_TYPES,
    display_name_for_slug,
    load_rfc_values_by_group_streaming,
    load_rfc_values_streaming,
    strategy_2_slug,
)


def _escape_tex(text: str) -> str:
    return text.replace("_", r"\_")


def _fmt_num(v: Any, decimals: int) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(f):
        return "--"
    return f"{f:.{decimals}f}"


def _bias_key_match(d: Dict[Any, Any], bias: float) -> Any:
    for k in d.keys():
        try:
            if float(k) == float(bias):
                return k
        except (TypeError, ValueError):
            continue
    return bias


def _canonical_slug_order(slug2algo: Dict[str, str]) -> List[str]:
    preferred = ["H0", "H1", "A0", "A1", "A2", "S0", "S1"]
    present = [s for s in preferred if s in slug2algo]
    rest = [s for s in sorted(slug2algo.keys()) if s not in present]
    return present + rest


def _build_table(
    windows: List[int],
    biases: List[float],
    per_w_data: Dict[int, Dict[str, Dict[Any, float]]],
    slug2algo: Dict[str, str],
    caption: str,
    label: str,
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
    row_values: Dict[str, List[Any]] = {}
    for slug in ordered_slugs:
        algo = slug2algo[slug]
        vals: List[Any] = []
        for w in windows:
            by_bias = (per_w_data.get(w, {}) or {}).get(algo, {}) or {}
            for b in biases:
                key = _bias_key_match(by_bias, b)
                vals.append(by_bias.get(key, math.nan))
        row_values[slug] = vals

    # Per-column ranking: best = minimum RFC, second-best = second distinct minimum.
    # Missing values (NaN) are ignored.
    n_metric_cols = len(windows) * block_cols
    best_vals: List[Any] = [None] * n_metric_cols
    second_vals: List[Any] = [None] * n_metric_cols
    for col in range(n_metric_cols):
        col_nums: List[float] = []
        for slug in ordered_slugs:
            v = row_values[slug][col]
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(f):
                continue
            col_nums.append(f)
        if not col_nums:
            continue
        uniq_sorted = sorted(set(col_nums))
        best_vals[col] = uniq_sorted[0]
        if len(uniq_sorted) > 1:
            second_vals[col] = uniq_sorted[1]

    for slug in ordered_slugs:
        row_vals: List[str] = []
        for col, raw in enumerate(row_values[slug]):
            s = _fmt_num(raw, decimals=decimals)
            if s == "--":
                row_vals.append(s)
                continue
            try:
                f = float(raw)
            except (TypeError, ValueError):
                row_vals.append(s)
                continue
            if best_vals[col] is not None and f == best_vals[col]:
                row_vals.append(rf"\textbf{{{s}}}")
            elif second_vals[col] is not None and f == second_vals[col]:
                row_vals.append(rf"\underline{{{s}}}")
            else:
                row_vals.append(s)
        label = display_name_for_slug(slug2algo, slug)
        lines.append(rf"{_escape_tex(label)} & " + " & ".join(row_vals) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _build_table_vertical(
    windows: List[int],
    biases: List[float],
    per_w_data: Dict[int, Dict[str, Dict[Any, float]]],
    slug2algo: Dict[str, str],
    caption: str,
    label: str,
    decimals: int,
    *,
    algo_col_width_cm: float = 3.2,
) -> str:
    """
    Kompaktní „vertikální“ layout: pro každé $W$ blok řádků (multirow v prvním sloupci),
    druhý sloupec algoritmus, dále sloupce $\\beta=\\ldots$.
    Pořadí jako v textu závěru: ``\\centering`` → ``\\small`` → ``\\caption`` → ``tabular`` → ``\\label``.
    Vyžaduje v preambuli: ``\\usepackage{multirow}``.
    """
    ordered_slugs = _canonical_slug_order(slug2algo)
    n_slugs = len(ordered_slugs)
    n_biases = len(biases)

    row_values: Dict[str, Dict[int, List[Any]]] = {}
    for wi, w in enumerate(windows):
        for slug in ordered_slugs:
            algo = slug2algo[slug]
            by_bias = (per_w_data.get(w, {}) or {}).get(algo, {}) or {}
            if slug not in row_values:
                row_values[slug] = {}
            vals: List[Any] = []
            for b in biases:
                key = _bias_key_match(by_bias, b)
                vals.append(by_bias.get(key, math.nan))
            row_values[slug][wi] = vals

    best_vals: Dict[tuple, Any] = {}
    second_vals: Dict[tuple, Any] = {}
    for wi in range(len(windows)):
        for bj in range(n_biases):
            col_nums: List[float] = []
            for slug in ordered_slugs:
                v = row_values[slug][wi][bj]
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(f):
                    continue
                col_nums.append(f)
            if not col_nums:
                continue
            uniq_sorted = sorted(set(col_nums))
            best_vals[(wi, bj)] = uniq_sorted[0]
            if len(uniq_sorted) > 1:
                second_vals[(wi, bj)] = uniq_sorted[1]

    col_spec = rf"c p{{{algo_col_width_cm:.1f}cm}} " + " ".join(["c"] * n_biases)
    beta_headers = " & ".join(rf"$\beta={b:.1f}$" for b in biases)

    # Stejné pořadí jako typická šablona v textu: caption nad tabulkou, \small, pak \label až za \end{tabular}.
    lines: List[str] = [
        r"% requires \usepackage{multirow}",
        r"\begin{table}",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"$\omega$ & \textbf{{Algoritmus}} & {beta_headers} \\",
        r"\midrule",
    ]

    for wi, w in enumerate(windows):
        for si, slug in enumerate(ordered_slugs):
            row_prefix = (
                rf"\multirow{{{n_slugs}}}{{*}}{{${w}$}} & "
                if si == 0
                else "& "
            )
            label_txt = _escape_tex(display_name_for_slug(slug2algo, slug))
            cells: List[str] = []
            for bj in range(n_biases):
                raw = row_values[slug][wi][bj]
                s = _fmt_num(raw, decimals=decimals)
                if s != "--":
                    try:
                        f = float(raw)
                        if not math.isnan(f):
                            if best_vals.get((wi, bj)) is not None and f == best_vals[(wi, bj)]:
                                s = rf"\textbf{{{s}}}"
                            elif second_vals.get((wi, bj)) is not None and f == second_vals[(wi, bj)]:
                                s = rf"\underline{{{s}}}"
                    except (TypeError, ValueError):
                        pass
                cells.append(s)
            lines.append(
                row_prefix + label_txt + " & " + " & ".join(cells) + r" \\"
            )
        if wi < len(windows) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _build_group_type_aggregate_table(
    windows: List[int],
    biases: List[float],
    per_w_group_data: Dict[int, Dict[str, Dict[str, Dict[Any, float]]]],
    group_types: List[str],
    caption: str,
    label: str,
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
        r"\textbf{group\_type} & " + " & ".join(header_biases_parts) + r" \\",
        r"\midrule",
    ]

    slug2algo = strategy_2_slug(ALGOS)
    canonical_algos = [slug2algo[s] for s in ["H0", "H1", "A0", "A1", "A2", "S0", "S1"] if s in slug2algo]

    for gt in group_types:
        row_vals: List[str] = []
        for w in windows:
            nested = per_w_group_data.get(w, {}) or {}
            for b in biases:
                vals: List[float] = []
                for algo in canonical_algos:
                    by_gt = (nested.get(algo) or {}).get(gt) or {}
                    key = _bias_key_match(by_gt, b)
                    raw = by_gt.get(key, math.nan)
                    try:
                        f = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(f):
                        continue
                    vals.append(f)
                row_vals.append(_fmt_num(sum(vals) / len(vals), decimals=decimals) if vals else "--")
        lines.append(rf"{_escape_tex(gt)} & " + " & ".join(row_vals) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print LaTeX table: default is horizontal ($W$ blocks × bias subcolumns); "
            "use --vertical for compact multirow layout (ω × algoritmus × β)."
        )
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="W",
        help="Window sizes shown as horizontal blocks (default: 1 3 5 10).",
    )
    parser.add_argument(
        "--biases",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0],
        metavar="B",
        help="Bias columns (default: 0 1 2).",
    )
    parser.add_argument(
        "--group-types",
        nargs="*",
        default=["similar", "outlier", "random"],
        metavar="GT",
        help="Group types used for averaging (default: similar outlier random).",
    )
    parser.add_argument(
        "--groups-count",
        type=int,
        default=None,
        help="Optional eval_n_<N> selector (e.g. 1000).",
    )
    parser.add_argument(
        "--latest-pickle-only",
        action="store_true",
        help="Load only latest pickle; default merges all numbered pickles.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Numeric precision for rendered values (default: 2).",
    )
    parser.add_argument(
        "--label",
        default="tab:rfc_by_bias_all_windows",
        help="LaTeX label (default: tab:rfc_by_bias_all_windows).",
    )
    parser.add_argument(
        "--label-group-type-aggregate",
        default="tab:rfc_group_type_aggregate_by_w_bias",
        help="LaTeX label for group-type aggregate table.",
    )
    parser.add_argument(
        "--no-group-type-aggregate",
        action="store_true",
        help="Do not print additional group-type aggregate table (mean across algorithms).",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help=(
            "Vertikální souhrnná tabulka (\\centering, \\small, \\caption nad tabulkou, "
            "$\\omega$ | Algoritmus | $\\beta$). Vyžaduje \\usepackage{multirow}. "
            "Kapitola: často \\texttt{--label tab:algos-comp}."
        ),
    )
    parser.add_argument(
        "--caption",
        default=None,
        metavar="TEX",
        help=(
            "Vlastní LaTeX text \\caption{...} jen pro --vertical (bez vnějších složených závorek). "
            "Výchozí: krátký automatický popis podle metriky."
        ),
    )
    parser.add_argument(
        "--algo-col-width",
        type=float,
        default=3.2,
        metavar="CM",
        help="Šířka sloupce s názvem algoritmu v --vertical režimu (p{CMcm}, výchozí 3.2).",
    )
    add_rfc_metric_arg(parser)
    args = parser.parse_args()
    if args.groups_count is None:
        print(
            "[warn] --groups-count is not set; loader may use non-strict fallback paths "
            "(can mix results from different eval_n_* runs).",
            file=sys.stderr,
        )

    spec = resolve_rfc_metric(args.rfc_metric)
    slug2algo = strategy_2_slug(ALGOS)
    merge = not args.latest_pickle_only
    group_types_eff = list(args.group_types) if args.group_types else list(GROUP_TYPES)

    per_w_data: Dict[int, Dict[str, Dict[Any, float]]] = {}
    per_w_group_data: Dict[int, Dict[str, Dict[str, Dict[Any, float]]]] = {}
    for w in args.windows:
        per_w_data[w] = load_rfc_values_streaming(
            ALGOS,
            str(w),
            metric_key=spec.storage_key,
            eval_type=EVAL_TYPE,
            groups_count=args.groups_count,
            merge_all_pickles=merge,
            group_types=group_types_eff,
        )
        per_w_group_data[w] = load_rfc_values_by_group_streaming(
            ALGOS,
            str(w),
            metric_key=spec.storage_key,
            eval_type=EVAL_TYPE,
            groups_count=args.groups_count,
            merge_all_pickles=merge,
            group_types=group_types_eff,
        )

    caption = (
        rf"Porovnání algoritmů skrze population bias pro více velikostí okna "
        rf"({spec.latex_caption_cs}); pro každé $W$ jsou uvedeny biasy {', '.join(str(b) for b in args.biases)}."
    )
    if args.vertical:
        caption_vert = caption if args.caption is None else args.caption
        print(
            _build_table_vertical(
                windows=args.windows,
                biases=list(args.biases),
                per_w_data=per_w_data,
                slug2algo=slug2algo,
                caption=caption_vert,
                label=args.label,
                decimals=args.decimals,
                algo_col_width_cm=args.algo_col_width,
            )
        )
    else:
        print(
            _build_table(
                windows=args.windows,
                biases=list(args.biases),
                per_w_data=per_w_data,
                slug2algo=slug2algo,
                caption=caption,
                label=args.label,
                decimals=args.decimals,
            )
        )
    if not args.no_group_type_aggregate:
        print()
        print(
            _build_group_type_aggregate_table(
                windows=args.windows,
                biases=list(args.biases),
                per_w_group_data=per_w_group_data,
                group_types=group_types_eff,
                caption=(
                    rf"Průměr přes algoritmy po typech skupin; stejné osy $W \times$ bias "
                    rf"({spec.latex_caption_cs})."
                ),
                label=args.label_group_type_aggregate,
                decimals=args.decimals,
            )
        )

