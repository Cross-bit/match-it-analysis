"""
Vizualizace success rate (poměr matched / total přes vybrané typy skupin) vůči population bias.

Osa x: bias beta (typicky tri body 0, 1, 2).
Osa y: success ratio v [0, 1].
Kazda krivka: jeden algoritmus (stejna agregace jako ``table_success_matches_all_windows``).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood import (
    ALGOS,
    EVAL_TYPE,
    order_algo_modules_paper,
    short_name_from_algo,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_success_matches_all_windows import (
    _bias_get,
    _load_per_algo_group_bias,
)

PALETTE_BY_NAME = {
    "tab10": "tab10",
    "tab10_soft": "custom_soft",
    "dark2": "Dark2",
    "set2": "Set2",
}

SOFT_TAB10_COLORS = [
    "#5b8cc0",
    "#e08b73",
    "#8dbf8a",
    "#b08fca",
    "#bfb27a",
    "#9ec1c8",
    "#d992b4",
]

# Font sizing aligned with other two-panel publication plots.
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11.5
LEGEND_TITLE_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 14

# Keep this plot aligned with paper algorithms currently evaluated in cache.
# STS prototype is intentionally excluded from plotting outputs for now.
PLOT_ALGOS = [a for a in ALGOS if a != "eval_sts_group_dynamic.py"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_out_path(windows: Sequence[int], output_format: str) -> Path:
    if len(windows) == 1:
        stem = f"success_rate_w{windows[0]}_bias"
    else:
        stem = "success_rate_all_windows_bias"
    return _repo_root() / "docs" / "img" / f"{stem}.{output_format}"


def success_ratios_for_algo(
    algo_name: str,
    w_size: int,
    biases: Sequence[float],
    group_types: List[str],
    eval_type: str,
    groups_count: int | None,
    merge_all_pickles: bool,
) -> List[float]:
    """Stejna agregace jako radky v ``table_success_matches_all_windows`` (soucet pres group types)."""
    try:
        per_gt = _load_per_algo_group_bias(
            algo_name,
            w_size,
            eval_type=eval_type,
            groups_count=groups_count,
            group_types=group_types,
            merge_all_pickles=merge_all_pickles,
        )
    except FileNotFoundError:
        # Some repos include optional algorithms in ALGOS without computed cache yet.
        # Keep plotting robust by returning NaNs for missing algorithms.
        return [float("nan")] * len(biases)
    out: List[float] = []
    for b in biases:
        matched_sum = 0.0
        total_sum = 0.0
        any_data = False
        for gt in group_types:
            pair = _bias_get(per_gt.get(gt, {}), float(b))
            if not pair:
                continue
            m, t = pair
            if math.isfinite(m) and math.isfinite(t) and t > 0:
                matched_sum += m
                total_sum += t
                any_data = True
        if any_data and total_sum > 0:
            out.append(matched_sum / total_sum)
        else:
            out.append(float("nan"))
    return out


def _plot_one_axis(
    *,
    ax,
    window_size: int,
    biases: Sequence[float],
    group_types: List[str],
    eval_type: str,
    groups_count: int | None,
    merge_all_pickles: bool,
    algos: Sequence[str],
    palette: str,
    use_czech: bool,
    legend_inside: bool = True,
    precomputed_by_algo: dict[str, list[float]] | None = None,
) -> None:
    ordered = order_algo_modules_paper(list(algos))
    x = np.asarray(biases, dtype=float)
    palette_kind = PALETTE_BY_NAME[palette]
    cmap = plt.get_cmap(palette_kind) if palette_kind != "custom_soft" else None

    for i, algo_name in enumerate(ordered):
        if precomputed_by_algo is not None:
            ys = precomputed_by_algo.get(algo_name, [float("nan")] * len(biases))
        else:
            ys = success_ratios_for_algo(
                algo_name,
                window_size,
                biases,
                group_types,
                eval_type,
                groups_count,
                merge_all_pickles,
            )
        y = np.asarray(ys, dtype=float)
        label = short_name_from_algo(algo_name)
        color = (
            SOFT_TAB10_COLORS[i % len(SOFT_TAB10_COLORS)]
            if palette_kind == "custom_soft"
            else cmap(i % 10)
        )
        ax.plot(x, y, "-o", color=color, linewidth=2.0, markersize=8, label=label, zorder=2)

    if use_czech:
        ax.set_xlabel(r"Bias populace $\beta$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Úspěšnost shody (0–1)", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_xlabel(r"Population bias $\beta$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Success ratio", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=1)
    if len(biases) <= 5:
        ax.set_xticks(list(biases))
    if legend_inside:
        ax.legend(
            loc="best",
            fontsize=LEGEND_FONTSIZE,
            ncol=2,
            framealpha=0.92,
            title_fontsize=LEGEND_TITLE_FONTSIZE,
        )


def plot_success_rate_curves(
    *,
    windows: Sequence[int],
    biases: Sequence[float],
    group_types: List[str],
    eval_type: str,
    groups_count: int | None,
    merge_all_pickles: bool,
    algos: Sequence[str],
    title: str | None,
    output: Path,
    show: bool,
    dpi: int,
    palette: str = "tab10",
    layout: str = "row",
    use_czech: bool = True,
    comparison_mode: str = "normal",
) -> None:
    # Keep original look for single-window plot; for many windows create side-by-side panels.
    windows_int = [int(w) for w in windows]
    if comparison_mode == "w1_vs_avg_rest":
        if 1 not in windows_int:
            raise ValueError("comparison_mode=w1_vs_avg_rest requires W=1 in --windows.")
        rest_ws = [w for w in windows_int if w != 1]
        if not rest_ws:
            raise ValueError("comparison_mode=w1_vs_avg_rest requires at least one non-1 window.")

        ordered = order_algo_modules_paper(list(algos))
        w1_values: dict[str, list[float]] = {}
        avg_rest_values: dict[str, list[float]] = {}
        for algo_name in ordered:
            w1_values[algo_name] = success_ratios_for_algo(
                algo_name, 1, biases, group_types, eval_type, groups_count, merge_all_pickles
            )
            stacked = []
            for w in rest_ws:
                stacked.append(
                    success_ratios_for_algo(
                        algo_name, w, biases, group_types, eval_type, groups_count, merge_all_pickles
                    )
                )
            arr = np.asarray(stacked, dtype=float)
            avg_rest_values[algo_name] = (
                np.nanmean(arr, axis=0).tolist() if np.isfinite(arr).any() else [float("nan")] * len(biases)
            )

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), layout="constrained", sharey=True)
        ax1, ax2 = axes
        _plot_one_axis(
            ax=ax1,
            window_size=1,
            biases=biases,
            group_types=group_types,
            eval_type=eval_type,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
            algos=algos,
            palette=palette,
            use_czech=use_czech,
            legend_inside=True,
            precomputed_by_algo=w1_values,
        )
        _plot_one_axis(
            ax=ax2,
            window_size=rest_ws[0],
            biases=biases,
            group_types=group_types,
            eval_type=eval_type,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
            algos=algos,
            palette=palette,
            use_czech=use_czech,
            legend_inside=True,
            precomputed_by_algo=avg_rest_values,
        )
        ax1.set_title(r"$\omega$=1", fontsize=PANEL_TITLE_FONTSIZE)
        ax2.set_title(rf"AVG($\omega$={','.join(str(w) for w in rest_ws)})", fontsize=PANEL_TITLE_FONTSIZE)
    elif len(windows) == 1:
        window_size = int(windows[0])
        fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
        _plot_one_axis(
            ax=ax,
            window_size=window_size,
            biases=biases,
            group_types=group_types,
            eval_type=eval_type,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
            algos=algos,
            palette=palette,
            use_czech=use_czech,
            legend_inside=True,
        )
        tit = title
        if tit is None:
            gc = groups_count if groups_count is not None else "mixed"
            tit = (
                (f"Úspěšnost shody vs. bias (agregace: {', '.join(group_types)}; " if use_czech else
                 f"Success rate vs bias (aggregated: {', '.join(group_types)}; ")
                + rf"$\omega$={window_size}; groups_count={gc})"
            )
        ax.set_title(tit, fontsize=11)
    else:
        n = len(windows_int)
        if layout == "grid2":
            rows = 2
            cols = int(math.ceil(n / 2))
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(5.2 * cols, 4.6 * rows),
                layout="constrained",
                sharey=True,
            )
            axes_flat = np.asarray(axes, dtype=object).ravel()
        else:
            fig, axes = plt.subplots(
                1,
                n,
                figsize=(5.2 * n, 4.8),
                layout="constrained",
                sharey=True,
            )
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes], dtype=object)
            axes_flat = axes

        for ax, w in zip(axes_flat, windows_int):
            _plot_one_axis(
                ax=ax,
                window_size=w,
                biases=biases,
                group_types=group_types,
                eval_type=eval_type,
                groups_count=groups_count,
                merge_all_pickles=merge_all_pickles,
                algos=algos,
                palette=palette,
                use_czech=use_czech,
                legend_inside=True,  # keep legend in each panel for now
            )
            ax.set_title(rf"$\omega$={w}", fontsize=11)
        for ax in axes_flat[len(windows_int):]:
            ax.axis("off")

        if title is None:
            gc = groups_count if groups_count is not None else "mixed"
            fig.suptitle(
                (
                    f"Úspěšnost shody vs. bias (agregace: {', '.join(group_types)}; groups_count={gc})"
                    if use_czech
                    else f"Success rate vs bias (aggregated: {', '.join(group_types)}; groups_count={gc})"
                ),
                fontsize=12,
            )
        else:
            fig.suptitle(title, fontsize=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot success ratio [0,1] vs bias (one curve per algorithm)."
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Single consensus window W. If omitted and --windows not set, plots all default windows.",
    )
    p.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=None,
        help="Multiple windows for side-by-side panels (e.g. --windows 1 3 5 10).",
    )
    p.add_argument(
        "--biases",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="Bias values on x-axis (typically three points).",
    )
    p.add_argument(
        "--group-types",
        nargs="*",
        default=["similar", "outlier", "random"],
        help="Same aggregation as success-matches table.",
    )
    p.add_argument("--eval-type", default=EVAL_TYPE, choices=["train", "validation", "test"])
    p.add_argument("--groups-count", type=int, default=None)
    p.add_argument("--latest-pickle-only", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output path. Extension decides format (png/pdf/...).",
    )
    p.add_argument(
        "--output-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Default format used when --output is not provided (default: pdf).",
    )
    p.add_argument("--title", default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
    p.add_argument(
        "--palette",
        choices=sorted(PALETTE_BY_NAME.keys()),
        default="tab10",
        help="Color palette variant for curves.",
    )
    p.add_argument(
        "--layout",
        choices=["row", "grid2"],
        default="row",
        help="Multi-window arrangement: row (1xN) or grid2 (2xceil(N/2)).",
    )
    p.add_argument(
        "--comparison-mode",
        choices=["normal", "w1_vs_avg_rest"],
        default="normal",
        help="normal: one panel per W. w1_vs_avg_rest: two panels (W=1 and average over remaining W).",
    )
    p.add_argument(
        "--english",
        action="store_true",
        help="Axis labels, legend and default title in English (default is Czech).",
    )
    args = p.parse_args()

    if args.windows is not None and args.window_size is not None:
        raise ValueError("Use either --window-size or --windows, not both.")
    if args.windows is not None:
        windows = [int(w) for w in args.windows]
    elif args.window_size is not None:
        windows = [int(args.window_size)]
    else:
        windows = [1, 3, 5, 10]

    out = args.output if args.output is not None else _default_out_path(windows, args.output_format)
    gt = list(args.group_types) if args.group_types else ["similar", "outlier", "random"]

    plot_success_rate_curves(
        windows=windows,
        biases=list(args.biases),
        group_types=gt,
        eval_type=args.eval_type,
        groups_count=args.groups_count,
        merge_all_pickles=not args.latest_pickle_only,
        algos=PLOT_ALGOS,
        title=args.title,
        output=out,
        show=args.show,
        dpi=args.dpi,
        palette=args.palette,
        layout=args.layout,
        use_czech=not args.english,
        comparison_mode=args.comparison_mode,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
