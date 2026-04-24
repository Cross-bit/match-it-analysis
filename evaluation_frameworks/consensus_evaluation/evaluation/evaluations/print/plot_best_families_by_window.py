import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res


ASYNC_ALGOS = [
    "eval_async_static_policy_simple_priority_function_group_rec.py",
    "eval_async_static_policy_simple_priority_function_individual_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
]

HYBRID_ALGOS = [
    "eval_hybrid_general_rec_individual.py",
    "eval_hybrid_updatable.py",
]

SYNC_ALGOS = [
    "eval_sync_without_feedback.py",
    "eval_sync_with_feedback_ema.py",
]

DEFAULT_GROUP_TYPES = ["similar", "outlier", "random", "divergent", "variance"]


def _default_img_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    return repo_root / "docs" / "img"


def _bias_key_to_float(v: Any) -> Any:
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


def _pick_bias_block(group_data: Dict[Any, Any], bias: float) -> Dict[str, Any]:
    if not isinstance(group_data, dict):
        return {}
    target = float(bias)
    for k, v in group_data.items():
        if _bias_key_to_float(k) == target:
            return v if isinstance(v, dict) else {}
    return {}


def _safe_float(d: Dict[str, Any], key: str) -> float:
    if not isinstance(d, dict):
        return math.nan
    if key in d:
        try:
            return float(d[key])
        except (TypeError, ValueError):
            return math.nan
    metrics = d.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        try:
            return float(metrics[key])
        except (TypeError, ValueError):
            return math.nan
    return math.nan


def _metric_value(stats_block: Dict[str, Any], metric: str, w: int) -> float:
    if metric == "rounds_to_consensus":
        return _safe_float(stats_block, "average")
    if metric == "cards_seen_until_consensus":
        direct = _safe_float(stats_block, "first_consensus_global_position_across_groups")
        if not math.isnan(direct):
            return direct
        avg_round = _safe_float(stats_block, "average")
        rank = _safe_float(stats_block, "first_consensus_rank_across_groups")
        if not math.isnan(avg_round) and not math.isnan(rank):
            return float(w) * (avg_round - 1.0) + rank
        return math.nan
    raise ValueError(f"Unknown metric: {metric}")


def _algo_mean_for_window(
    algo_name: str,
    w: int,
    *,
    bias: float,
    group_types: List[str],
    metric: str,
    groups_count: int | None,
) -> float:
    try:
        data = load_eval_res(
            algo_name,
            str(w),
            "test",
            groups_count=groups_count,
            merge_all_pickles=True,
        )
    except Exception as e:
        print(f"  [warn] load failed for {algo_name} @ W={w}: {e}")
        return math.nan

    vals: List[float] = []
    for gt in group_types:
        gt_block = data.get(gt, {})
        stats = _pick_bias_block(gt_block, bias)
        val = _metric_value(stats, metric, w)
        if not math.isnan(val):
            vals.append(val)
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def _best_family_for_window(
    family_algos: List[str],
    w: int,
    *,
    bias: float,
    group_types: List[str],
    metric: str,
    groups_count: int | None,
) -> Tuple[str, float]:
    best_algo = ""
    best_val = math.nan
    for algo in family_algos:
        val = _algo_mean_for_window(
            algo,
            w,
            bias=bias,
            group_types=group_types,
            metric=metric,
            groups_count=groups_count,
        )
        if math.isnan(val):
            continue
        if math.isnan(best_val) or val < best_val:
            best_val = val
            best_algo = algo
    return best_algo, best_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot best async/hybrid/sync family metric by window size."
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument(
        "--groups-count",
        type=int,
        default=None,
        help="Optional eval_n_<N> selector. Omit for auto-discovery across cache layouts.",
    )
    parser.add_argument("--group-types", nargs="+", default=DEFAULT_GROUP_TYPES)
    parser.add_argument(
        "--metric",
        choices=["rounds_to_consensus", "cards_seen_until_consensus"],
        default="rounds_to_consensus",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path (.png preferred). If omitted, saves to <repo>/docs/img/.",
    )
    parser.add_argument(
        "--img-dir",
        default=None,
        help="Directory used when --output is omitted (default: <repo>/docs/img).",
    )
    parser.add_argument(
        "--output-stem",
        default="best_families_by_window",
        help="Base filename (without extension) when --output is omitted.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--show", action="store_true", help="Show figure window (default on).")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window.")
    args = parser.parse_args()

    windows = sorted(set(args.windows))
    family_specs = [
        ("Async (best of A0-A2)", ASYNC_ALGOS, "tab:blue"),
        ("Hybrid (best of H0-H1)", HYBRID_ALGOS, "tab:green"),
        ("Sync (best of S0-S1)", SYNC_ALGOS, "tab:red"),
    ]

    print("=== Best algorithm per family and window ===")
    series: Dict[str, List[float]] = {}
    for family_name, algos, _ in family_specs:
        ys: List[float] = []
        print(f"\n[{family_name}]")
        for w in windows:
            best_algo, best_val = _best_family_for_window(
                algos,
                w,
                bias=args.bias,
                group_types=args.group_types,
                metric=args.metric,
                groups_count=args.groups_count,
            )
            ys.append(best_val)
            print(f"  W={w}: {best_algo or 'N/A'} -> {best_val}")
        series[family_name] = ys

    plt.figure(figsize=(9, 5))
    for family_name, _, color in family_specs:
        plt.plot(windows, series[family_name], marker="o", linewidth=2, color=color, label=family_name)

    metric_label = (
        "RFC (rounds_to_consensus)"
        if args.metric == "rounds_to_consensus"
        else "Cards seen until consensus"
    )
    title = args.title or f"Best family by window | metric={args.metric}, bias={args.bias}"
    plt.title(title)
    plt.xlabel("Window size (W)")
    plt.ylabel(metric_label)
    # Show only the evaluated window sizes on x-axis.
    plt.xticks(windows)
    if windows:
        pad = 0.3 if len(windows) == 1 else 0.2 * (max(windows) - min(windows)) / max(1, len(windows) - 1)
        plt.xlim(min(windows) - pad, max(windows) + pad)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if args.output:
        out_png = Path(args.output)
        if out_png.suffix.lower() != ".png":
            out_png = out_png.with_suffix(".png")
        out_pdf = out_png.with_suffix(".pdf")
    else:
        img_dir = Path(args.img_dir) if args.img_dir else _default_img_dir()
        img_dir.mkdir(parents=True, exist_ok=True)
        out_png = img_dir / f"{args.output_stem}.png"
        out_pdf = img_dir / f"{args.output_stem}.pdf"

    plt.savefig(out_png, dpi=170)
    plt.savefig(out_pdf)
    print(f"\nSaved plot PNG to: {out_png.resolve()}")
    print(f"Saved plot PDF to: {out_pdf.resolve()}")
    want_show = (not args.no_show) or args.show
    has_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if want_show and has_gui:
        plt.show()
    elif want_show and not has_gui:
        print("GUI display not available (no DISPLAY/WAYLAND_DISPLAY); skipping plt.show().")


if __name__ == "__main__":
    main()

