from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils.config import CACHE_FILES_DIR, load_from_pickle


DEFAULT_ALGO_SLUGS: Dict[str, str] = {
    "eval_async_static_policy_simple_priority_function_group_rec.py": "A0",
    "eval_async_static_policy_simple_priority_function_individual_rec.py": "A1",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py": "A2",
    "eval_sts_group_dynamic.py": "A3",
    "eval_hybrid_general_rec_individual.py": "H0",
    "eval_hybrid_updatable.py": "H1",
    "eval_sync_without_feedback.py": "S0",
    "eval_sync_with_feedback_ema.py": "S1",
    # large-group module names mapped to same canonical families
    "eval_large_hybrid_general_rec_individual.py": "H0",
    "eval_large_hybrid_group_updatable.py": "H1",
    "eval_large_sync_without_feedback.py": "S0",
    "eval_large_sync_with_feedback_ema.py": "S1",
}

SHORT_NAME_BY_SLUG: Dict[str, str] = {
    "A0": "Async-Static-Grp",
    "A1": "Async-Static-Ind",
    "A2": "Async-Dyn-Ind",
    "A3": "STS-Dyn",
    "H0": "Hybrid-Ind",
    "H1": "Hybrid-Grp-EMA",
    "S0": "Sync",
    "S1": "Sync-EMA",
}


def _algo_slug(algo_name: str) -> str:
    if algo_name in DEFAULT_ALGO_SLUGS:
        return DEFAULT_ALGO_SLUGS[algo_name]
    if "async" in algo_name:
        return "A?"
    if "hybrid" in algo_name:
        return "H?"
    if "sync" in algo_name:
        return "S?"
    return "?"


def _algo_family(algo_name: str) -> str:
    slug = _algo_slug(algo_name)
    if slug.startswith("A"):
        return "async"
    if slug.startswith("H"):
        return "hybrid"
    if slug.startswith("S"):
        return "sync"
    return "other"


def _algo_short_name(algo_name: str) -> str:
    slug = _algo_slug(algo_name)
    return SHORT_NAME_BY_SLUG.get(slug, algo_name.replace(".py", ""))


def _is_numbered_pickle(path: Path) -> bool:
    if path.suffix.lower() != ".pkl":
        return False
    try:
        int(path.stem)
        return True
    except ValueError:
        return False


def _parse_path_metadata_labeled(parts: Tuple[str, ...], pkl_path: Path) -> Optional[Dict[str, Any]]:
    """
    Expected layout:
      cache/cons_evaluations/w_<W>/[group_<n>/]split_<eval>/[eval_n_<k>/]<algo>.py/<N>.pkl
    """
    if len(parts) < 4:
        return None

    if not parts[0].startswith("w_"):
        return None
    w_size = parts[0][2:]

    idx = 1
    group_size: Optional[int] = None
    if idx < len(parts) and parts[idx].startswith("group_"):
        try:
            group_size = int(parts[idx][6:])
        except ValueError:
            group_size = None
        idx += 1

    if idx >= len(parts) or not parts[idx].startswith("split_"):
        return None
    eval_type = parts[idx][6:]
    idx += 1

    groups_count: Optional[int] = None
    if idx < len(parts) and parts[idx].startswith("eval_n_"):
        try:
            groups_count = int(parts[idx][7:])
        except ValueError:
            groups_count = None
        idx += 1

    if idx >= len(parts):
        return None
    algo_name = parts[idx]

    try:
        run_num = int(pkl_path.stem)
    except ValueError:
        return None

    return {
        "w_size": int(w_size) if str(w_size).isdigit() else w_size,
        "eval_type": eval_type,
        "group_size": group_size,
        "groups_count": groups_count,
        "algorithm_file": algo_name,
        "algorithm_slug": _algo_slug(algo_name),
        "algorithm_short_name": _algo_short_name(algo_name),
        "algorithm_family": _algo_family(algo_name),
        "run_num": run_num,
        "cache_path": str(pkl_path),
        "cache_layout": "labeled",
    }


def _parse_path_metadata_legacy(parts: Tuple[str, ...], pkl_path: Path) -> Optional[Dict[str, Any]]:
    """
    Legacy layout:
      cache/cons_evaluations/<W>/[large/<group_size>/]<split>/<algo>.py/<N>.pkl
    """
    if len(parts) < 4:
        return None
    if parts[0].startswith("w_"):
        return None

    w_size_raw = parts[0]
    try:
        w_size = int(w_size_raw)
    except ValueError:
        w_size = w_size_raw

    idx = 1
    group_size: Optional[int] = None
    if idx < len(parts) and parts[idx] == "large":
        if idx + 1 >= len(parts):
            return None
        try:
            group_size = int(parts[idx + 1])
        except ValueError:
            group_size = None
        idx += 2

    if idx >= len(parts):
        return None
    eval_type = parts[idx]
    idx += 1
    if idx >= len(parts):
        return None
    algo_name = parts[idx]

    try:
        run_num = int(pkl_path.stem)
    except ValueError:
        return None

    return {
        "w_size": w_size,
        "eval_type": eval_type,
        "group_size": group_size,
        "groups_count": None,  # legacy layout has no eval_n_<N> segment
        "algorithm_file": algo_name,
        "algorithm_slug": _algo_slug(algo_name),
        "algorithm_short_name": _algo_short_name(algo_name),
        "algorithm_family": _algo_family(algo_name),
        "run_num": run_num,
        "cache_path": str(pkl_path),
        "cache_layout": "legacy",
    }


def _parse_path_metadata(pkl_path: Path, cache_root: Path) -> Optional[Dict[str, Any]]:
    try:
        rel = pkl_path.relative_to(cache_root)
    except ValueError:
        return None
    parts = rel.parts
    meta = _parse_path_metadata_labeled(parts, pkl_path)
    if meta is not None:
        return meta
    return _parse_path_metadata_legacy(parts, pkl_path)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return math.nan


def _extract_scalar_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ("average", "variance", "std_dev"):
        if k in stats:
            out[k] = _safe_float(stats.get(k))

    # Optional precomputed global position metric from newer caches
    if "first_consensus_global_position_across_groups" in stats:
        out["first_consensus_global_position_across_groups"] = _safe_float(
            stats.get("first_consensus_global_position_across_groups")
        )

    # Pull summarized NDCG blocks, if available
    for key, val in stats.items():
        if not (isinstance(key, str) and key.startswith("ndcg@") and isinstance(val, dict)):
            continue
        suffix = key.replace("@", "_")
        out[f"{suffix}_per_user_ndcg_mean_overall"] = _safe_float(val.get("per_user_ndcg_mean_overall"))
        out[f"{suffix}_ndcg_com_mean_overall"] = _safe_float(val.get("ndcg_com_mean_overall"))

    # Success-rate related fields used by RFC-vs-success tradeoff plots.
    total = _safe_float(stats.get("total_rounds"))
    matched_obj = stats.get("matched_groups")
    matched_count = math.nan
    if isinstance(matched_obj, (list, tuple)):
        matched_count = float(len(matched_obj))
    elif isinstance(matched_obj, (int, float)):
        matched_count = float(matched_obj)
    elif hasattr(matched_obj, "__len__") and not isinstance(matched_obj, (str, bytes, dict)):
        try:
            matched_count = float(len(matched_obj))
        except Exception:
            matched_count = math.nan
    out["total_rounds"] = total
    out["matched_count"] = matched_count
    if math.isfinite(total) and total > 0 and math.isfinite(matched_count):
        out["success_rate"] = matched_count / total
    else:
        out["success_rate"] = math.nan
    return out


def _iter_group_bias_rows(payload: Dict[str, Any]) -> Iterable[Tuple[str, Any, Dict[str, Any]]]:
    """
    Payload shape is typically:
      { group_type: { bias: stats_dict } }
    """
    if not isinstance(payload, dict):
        return
    for group_type, bias_map in payload.items():
        if not isinstance(group_type, str) or not isinstance(bias_map, dict):
            continue
        for bias_key, stats in bias_map.items():
            if isinstance(stats, dict):
                yield group_type, bias_key, stats


def build_dataframe(cache_root: Path, latest_only: bool) -> pd.DataFrame:
    pkl_files = [p for p in cache_root.glob("**/*.pkl") if _is_numbered_pickle(p)]
    if not pkl_files:
        return pd.DataFrame()

    parsed: List[Tuple[Path, Dict[str, Any]]] = []
    for p in pkl_files:
        meta = _parse_path_metadata(p, cache_root)
        if meta is not None:
            parsed.append((p, meta))

    if latest_only:
        by_eval_dir: Dict[Path, Tuple[Path, Dict[str, Any]]] = {}
        for p, meta in parsed:
            key = p.parent
            current = by_eval_dir.get(key)
            if current is None or meta["run_num"] > current[1]["run_num"]:
                by_eval_dir[key] = (p, meta)
        parsed = list(by_eval_dir.values())

    rows: List[Dict[str, Any]] = []
    for p, meta in parsed:
        try:
            # Use absolute path so utils.config does not remap relative paths to CACHE_FILES_DIR.
            payload = load_from_pickle(str(p.resolve()))
        except Exception:
            continue

        for group_type, bias_key, stats in _iter_group_bias_rows(payload):
            row = dict(meta)
            row["group_type"] = group_type
            row["bias"] = _safe_float(bias_key)
            row["bias_raw"] = str(bias_key)
            row.update(_extract_scalar_metrics(stats))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    preferred = [
        "algorithm_slug",
        "algorithm_short_name",
        "algorithm_family",
        "algorithm_file",
        "w_size",
        "eval_type",
        "group_size",
        "groups_count",
        "cache_layout",
        "run_num",
        "group_type",
        "bias",
        "average",
        "success_rate",
        "matched_count",
        "total_rounds",
        "variance",
        "std_dev",
    ]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RFC evaluation cache into tidy pandas DataFrame (CSV/Parquet)."
    )
    parser.add_argument(
        "--cache-root",
        default=str(CACHE_FILES_DIR / "cons_evaluations"),
        help="Root directory with consensus evaluation cache (default: analysis/cache/cons_evaluations).",
    )
    parser.add_argument(
        "--out-csv",
        default=str(CACHE_FILES_DIR / "cons_evaluations" / "exports" / "rfc_results_flat.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-parquet",
        default=None,
        help="Optional Parquet path.",
    )
    parser.add_argument(
        "--out-pkl",
        default=None,
        help="Optional pickle path (pandas DataFrame via df.to_pickle).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Export all numbered pickles. Default is latest run per eval directory only.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="Optional filter by window sizes (e.g. --windows 1 3 5 10).",
    )
    parser.add_argument(
        "--biases",
        type=float,
        nargs="+",
        default=None,
        help="Optional filter by population biases (e.g. --biases 0 1 2).",
    )
    parser.add_argument(
        "--group-types",
        nargs="+",
        default=None,
        help="Optional filter by group types (e.g. --group-types similar outlier random).",
    )
    parser.add_argument(
        "--groups-counts",
        type=int,
        nargs="+",
        default=None,
        help="Optional filter by evaluated groups-count (e.g. --groups-counts 100).",
    )
    parser.add_argument(
        "--eval-type",
        default=None,
        help="Optional filter by eval split (e.g. validation or test).",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    out_csv = Path(args.out_csv)
    out_parquet = Path(args.out_parquet) if args.out_parquet else None
    out_pkl = Path(args.out_pkl) if args.out_pkl else None

    df = build_dataframe(cache_root=cache_root, latest_only=not args.all_runs)
    if df.empty:
        print(f"No valid eval rows found in {cache_root}")
        return

    if args.windows:
        df = df[df["w_size"].isin(args.windows)]
    if args.biases:
        df = df[df["bias"].isin(args.biases)]
    if args.group_types:
        df = df[df["group_type"].isin(args.group_types)]
    if args.groups_counts:
        df = df[df["groups_count"].isin(args.groups_counts)]
    if args.eval_type:
        df = df[df["eval_type"] == args.eval_type]

    if df.empty:
        print("No rows left after applying filters.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv} ({len(df)} rows)")

    if out_pkl is not None:
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_pickle(out_pkl)
            print(f"Saved PKL: {out_pkl}")
        except Exception as e:
            print(f"PKL export skipped ({e})")

    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"Saved Parquet: {out_parquet}")
        except Exception as e:
            print(f"Parquet export skipped ({e})")

    print("Columns:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    main()
