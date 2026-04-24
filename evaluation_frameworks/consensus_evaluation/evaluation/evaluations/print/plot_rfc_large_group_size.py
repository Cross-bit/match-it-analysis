"""
Plot RFC vs group size for large-group experiments.

- x-axis: group size
- y-axis: selected RFC metric
- one curve per algorithm
- optional multi-window tiling: row (1xN) or grid2 (2 x ceil(N/2))
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.rfc_table_metric_spec import (
    add_rfc_metric_arg,
    resolve_rfc_metric,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_large_group_size_comparisions import (
    ALGOS,
    _filter_algos_with_any_values,
    _short_name_from_algo,
    load_large_rfc_values,
)

EVAL_TYPE = "test"
PALETTE_BY_NAME = {
    "tab10": "tab10",
    "dark2": "Dark2",
    "set2": "Set2",
}

AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11.5
LEGEND_TITLE_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 14


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_out_path(windows: Sequence[int], output_format: str) -> Path:
    if len(windows) == 1:
        stem = f"rfc_large_group_size_w{windows[0]}"
    else:
        stem = "rfc_large_group_size_all_windows"
    return _repo_root() / "docs" / "img" / f"{stem}.{output_format}"


def _plot_one_axis(
    *,
    ax,
    values_by_algo: dict[str, dict[int, float]],
    algos: Sequence[str],
    group_sizes: Sequence[int],
    palette: str,
    apply_async_minus_one: bool,
    use_czech: bool,
    x_min: float | None,
) -> None:
    x = np.asarray(group_sizes, dtype=float)
    cmap = plt.get_cmap(PALETTE_BY_NAME[palette])
    for i, algo_name in enumerate(algos):
        ys = []
        for gs in group_sizes:
            val = values_by_algo.get(algo_name, {}).get(gs, float("nan"))
            if apply_async_minus_one and ("async" in algo_name) and pd.notna(val):
                val = float(val) - 1.0
            ys.append(val)
        label = _short_name_from_algo(algo_name)
        ax.plot(x, np.asarray(ys, dtype=float), "-o", color=cmap(i % 10), linewidth=2.0, markersize=7, label=label)

    ax.set_xticks(list(group_sizes))
    if x_min is not None:
        x_max = float(max(group_sizes)) + 0.2
        ax.set_xlim(float(x_min), x_max)
    ax.grid(True, linestyle="--", alpha=0.35)
    if use_czech:
        ax.set_xlabel("Počet členů skupiny")
    else:
        ax.set_xlabel("Group size (members)")


def plot_rfc_large_group_size(
    *,
    windows: Sequence[int],
    group_sizes: Sequence[int],
    groups_count: int,
    metric_storage_key: str,
    metric_caption_cs: str,
    subtract_one_for_async_slug: bool,
    only_available: bool,
    layout: str,
    palette: str,
    output: Path,
    dpi: int,
    show: bool,
    use_czech: bool,
    x_min: float | None,
    independent_y: bool,
    comparison_mode: str,
) -> None:
    windows_int = [int(w) for w in windows]

    per_window_values: dict[int, dict[str, dict[int, float]]] = {}
    per_window_algos: dict[int, list[str]] = {}
    for w in windows_int:
        vals = load_large_rfc_values(
            ALGOS,
            window_size=str(w),
            eval_type=EVAL_TYPE,
            groups_count=groups_count,
            group_sizes=list(group_sizes),
            metric_key=metric_storage_key,
        )
        algos_w = _filter_algos_with_any_values(vals, ALGOS) if only_available else list(ALGOS)
        per_window_values[w] = vals
        per_window_algos[w] = algos_w

    if comparison_mode == "w1_vs_avg_rest":
        if 1 not in windows_int:
            raise ValueError("comparison_mode=w1_vs_avg_rest requires W=1 in --windows.")
        rest_ws = [w for w in windows_int if w != 1]
        if not rest_ws:
            raise ValueError("comparison_mode=w1_vs_avg_rest requires at least one non-1 window.")

        algos_cmp = _filter_algos_with_any_values(
            {
                a: {
                    gs: np.nanmean(
                        [
                            per_window_values[w].get(a, {}).get(gs, float("nan"))
                            for w in [1] + rest_ws
                        ]
                    )
                    for gs in group_sizes
                }
                for a in ALGOS
            },
            ALGOS,
        ) if only_available else list(ALGOS)

        avg_rest_values: dict[str, dict[int, float]] = {}
        for algo_name in ALGOS:
            avg_rest_values[algo_name] = {}
            for gs in group_sizes:
                vals = [per_window_values[w].get(algo_name, {}).get(gs, float("nan")) for w in rest_ws]
                arr = np.asarray(vals, dtype=float)
                avg_rest_values[algo_name][gs] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

        # In two-panel comparison mode, keep independent y-scale per panel.
        # W=1 values are typically much larger, and shared scale visually flattens AVG(rest).
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), layout="constrained", sharey=False)
        ax1, ax2 = axes
        _plot_one_axis(
            ax=ax1,
            values_by_algo=per_window_values[1],
            algos=algos_cmp,
            group_sizes=group_sizes,
            palette=palette,
            apply_async_minus_one=subtract_one_for_async_slug,
            use_czech=use_czech,
            x_min=x_min,
        )
        _plot_one_axis(
            ax=ax2,
            values_by_algo=avg_rest_values,
            algos=algos_cmp,
            group_sizes=group_sizes,
            palette=palette,
            apply_async_minus_one=subtract_one_for_async_slug,
            use_czech=use_czech,
            x_min=x_min,
        )
        ax1.set_title(r"$\omega$=1", fontsize=PANEL_TITLE_FONTSIZE)
        ax2.set_title(rf"AVG($\omega$={','.join(str(w) for w in rest_ws)})", fontsize=PANEL_TITLE_FONTSIZE)
        if use_czech:
            ax1.set_ylabel("RFC", fontsize=AXIS_LABEL_FONTSIZE)
            ax2.set_ylabel("RFC", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax1.set_ylabel("RFC", fontsize=AXIS_LABEL_FONTSIZE)
            ax2.set_ylabel("RFC", fontsize=AXIS_LABEL_FONTSIZE)
        for ax in (ax1, ax2):
            ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax1.tick_params(axis="y", labelleft=True)
        ax2.tick_params(axis="y", labelleft=True)
        ax1.legend(
            loc="best",
            fontsize=LEGEND_FONTSIZE,
            ncol=1,
            framealpha=0.92,
            title_fontsize=LEGEND_TITLE_FONTSIZE,
        )
        ax2.legend(
            loc="best",
            fontsize=LEGEND_FONTSIZE,
            ncol=1,
            framealpha=0.92,
            title_fontsize=LEGEND_TITLE_FONTSIZE,
        )
    elif len(windows_int) == 1:
        w = windows_int[0]
        fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
        _plot_one_axis(
            ax=ax,
            values_by_algo=per_window_values[w],
            algos=per_window_algos[w],
            group_sizes=group_sizes,
            palette=palette,
            apply_async_minus_one=subtract_one_for_async_slug,
            use_czech=use_czech,
            x_min=x_min,
        )
        if use_czech:
            ax.set_ylabel("RFC")
            ax.set_title(
                f"Large-group: RFC vs počet členů skupiny ($\\omega$={w}, groups_count={groups_count})",
                fontsize=11,
            )
        else:
            ax.set_ylabel("RFC")
            ax.set_title(
                f"Large-group: RFC vs group size ($\\omega$={w}, groups_count={groups_count})",
                fontsize=11,
            )
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.92)
    else:
        n = len(windows_int)
        if layout == "grid2":
            rows = 2
            cols = int(np.ceil(n / 2))
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(5.3 * cols, 4.8 * rows),
                layout="constrained",
                sharey=not independent_y,
            )
            axes_flat = np.asarray(axes, dtype=object).ravel()
        else:
            fig, axes = plt.subplots(
                1,
                n,
                figsize=(5.3 * n, 4.9),
                layout="constrained",
                sharey=not independent_y,
            )
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes], dtype=object)
            axes_flat = axes
        if layout == "col":
            fig, axes = plt.subplots(
                n,
                1,
                figsize=(6.2, 4.3 * n),
                layout="constrained",
                sharey=not independent_y,
            )
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes], dtype=object)
            axes_flat = axes

        for ax, w in zip(axes_flat, windows_int):
            _plot_one_axis(
                ax=ax,
                values_by_algo=per_window_values[w],
                algos=per_window_algos[w],
                group_sizes=group_sizes,
                palette=palette,
                apply_async_minus_one=subtract_one_for_async_slug,
                use_czech=use_czech,
                x_min=x_min,
            )
            ax.set_title(rf"$\omega$={w}", fontsize=11)
            if use_czech:
                ax.set_ylabel("RFC")
            else:
                ax.set_ylabel("RFC")
            # Keep labels visible on every subplot.
            ax.tick_params(axis="y", labelleft=True)
            ax.legend(loc="best", fontsize=7.5, ncol=1, framealpha=0.92)

        for ax in axes_flat[len(windows_int):]:
            ax.axis("off")

        if use_czech:
            fig.suptitle(
                f"Large-group: {metric_caption_cs} podle velikosti skupiny (groups_count={groups_count})",
                fontsize=12,
            )
        else:
            fig.suptitle(
                f"Large-group RFC metric by group size (groups_count={groups_count})",
                fontsize=12,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=None, help="Single window W.")
    parser.add_argument("--windows", nargs="+", type=int, default=None, help="Multiple windows (for tiled plots).")
    parser.add_argument("--groups-count", type=int, default=1000)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--only-available", action="store_true")
    parser.add_argument("--layout", choices=["row", "grid2", "col"], default="row")
    parser.add_argument("--palette", choices=sorted(PALETTE_BY_NAME.keys()), default="tab10")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-format", choices=["pdf", "png"], default="pdf")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--english", action="store_true")
    parser.add_argument(
        "--x-min",
        type=float,
        default=4.5,
        help="Left x-axis limit (default 4.5 trims empty space before group size 5).",
    )
    parser.add_argument(
        "--independent-y",
        action="store_true",
        help="Use independent y-axis scale per subplot (recommended when W=1 has much larger values).",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=["normal", "w1_vs_avg_rest"],
        default="normal",
        help="normal: one panel per W. w1_vs_avg_rest: two panels (W=1 and average of remaining windows).",
    )
    add_rfc_metric_arg(parser)
    args = parser.parse_args()

    if args.windows is not None and args.window_size is not None:
        raise ValueError("Use either --window-size or --windows, not both.")
    if args.windows is not None:
        windows = [int(w) for w in args.windows]
    elif args.window_size is not None:
        windows = [int(args.window_size)]
    else:
        windows = [1, 3, 5, 10]

    spec = resolve_rfc_metric(args.rfc_metric)
    out = args.output if args.output is not None else _default_out_path(windows, args.output_format)
    plot_rfc_large_group_size(
        windows=windows,
        group_sizes=list(args.group_sizes),
        groups_count=args.groups_count,
        metric_storage_key=spec.storage_key,
        metric_caption_cs=spec.latex_caption_cs,
        subtract_one_for_async_slug=spec.subtract_one_for_async_slug,
        only_available=args.only_available,
        layout=args.layout,
        palette=args.palette,
        output=out,
        dpi=args.dpi,
        show=args.show,
        use_czech=not args.english,
        x_min=args.x_min,
        independent_y=args.independent_y,
        comparison_mode=args.comparison_mode,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
