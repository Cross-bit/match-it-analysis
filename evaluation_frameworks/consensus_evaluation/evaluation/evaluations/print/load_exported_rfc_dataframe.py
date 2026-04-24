from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


CANONICAL_SLUG_ORDER: list[str] = ["H0", "H1", "A0", "A1", "A2", "S0", "S1"]
CANONICAL_SLUG_SET = set(CANONICAL_SLUG_ORDER)
SHORT_NAME_BY_SLUG: dict[str, str] = {
    "A0": "Async-Static-Grp",
    "A1": "Async-Static-Ind",
    "A2": "Async-Dyn-Ind",
    "H0": "Hybrid-Ind",
    "H1": "Hybrid-Grp-EMA",
    "S0": "Sync",
    "S1": "Sync-EMA",
}


def load_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        try:
            return pd.read_pickle(path)
        except Exception as e:
            raise RuntimeError(
                "Failed to read .pkl via pandas. This usually means the pickle was created "
                "under a different Python/pandas environment (e.g. WSL venv vs Windows Python). "
                "Re-run this loader in the same environment that produced the pickle, or export "
                "and load the .csv instead."
            ) from e
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path} (expected .pkl/.csv/.parquet)")


def rfc_by_population_bias(df: pd.DataFrame, w: int, group_types: Optional[list[str]] = None) -> pd.DataFrame:
    d = df[(df["w_size"] == w) & (df["algorithm_slug"].isin(CANONICAL_SLUG_SET))].copy()
    if group_types is not None:
        d = d[d["group_type"].isin(group_types)]
    d = (
        d.groupby(["algorithm_slug", "bias"], as_index=False)["average"]
        .mean()
        .pivot(index="algorithm_slug", columns="bias", values="average")
    )
    d = d.reindex([s for s in CANONICAL_SLUG_ORDER if s in d.index])
    d = d.rename(index=lambda s: SHORT_NAME_BY_SLUG.get(str(s), str(s)))
    d.columns = [str(c) for c in d.columns]
    return d


def rfc_comparison(df: pd.DataFrame, w: int, bias: float, group_types: Optional[list[str]] = None) -> pd.DataFrame:
    d = df[
        (df["w_size"] == w)
        & (df["bias"] == bias)
        & (df["algorithm_slug"].isin(CANONICAL_SLUG_SET))
    ].copy()
    if group_types is not None:
        d = d[d["group_type"].isin(group_types)]
    d = (
        d.groupby(["algorithm_slug", "group_type"], as_index=False)["average"]
        .mean()
        .pivot(index="algorithm_slug", columns="group_type", values="average")
    )
    d = d.reindex([s for s in CANONICAL_SLUG_ORDER if s in d.index])
    d = d.rename(index=lambda s: SHORT_NAME_BY_SLUG.get(str(s), str(s)))
    return d


def _sorted_unique(series: pd.Series) -> list:
    vals = [v for v in series.dropna().unique().tolist()]
    try:
        return sorted(vals)
    except Exception:
        return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Load exported RFC dataframe and print basic pivots.")
    p.add_argument("--in", dest="inp", required=True, help="Path to exported dataframe (.pkl/.csv/.parquet).")
    p.add_argument("--head", type=int, default=5, help="Print N first rows preview.")
    p.add_argument("--w", type=int, default=None, help="Window size to build pivots for (e.g. 3).")
    p.add_argument("--bias", type=float, default=None, help="Bias for RFC comparison pivot (e.g. 0).")
    p.add_argument(
        "--all-windows",
        action="store_true",
        help="Print both table types for every window size present in the dataframe.",
    )
    p.add_argument(
        "--biases",
        type=float,
        nargs="+",
        default=None,
        help="Optional subset of biases to print comparisons for (e.g. --biases 0 1 2).",
    )
    p.add_argument(
        "--group-types",
        nargs="+",
        default=None,
        help="Optional subset of group types to use (e.g. similar outlier random).",
    )
    args = p.parse_args()

    df = load_df(Path(args.inp))
    print(f"Loaded df: {len(df)} rows × {len(df.columns)} cols")
    print("Columns:")
    print(", ".join(df.columns))
    print("")
    print("Head:")
    print(df.head(args.head).to_string(index=False))

    def _print_for_w(w: int) -> None:
        print("")
        print("=" * 72)
        print(f"W={w}")
        print("=" * 72)
        print("")
        print(f"RFC by population bias (w={w})")
        print(rfc_by_population_bias(df, w=w, group_types=args.group_types).to_string())

        biases = args.biases
        if biases is None:
            biases = _sorted_unique(df[df["w_size"] == w]["bias"])
        for b in biases:
            print("")
            print(f"RFC comparison (w={w}, bias={b})")
            print(rfc_comparison(df, w=w, bias=float(b), group_types=args.group_types).to_string())

    if args.all_windows:
        for w in _sorted_unique(df["w_size"]):
            try:
                w_int = int(w)
            except Exception:
                continue
            _print_for_w(w_int)
        return

    if args.w is not None:
        _print_for_w(int(args.w))
        return

    if args.bias is not None:
        print("")
        print("Note: --bias without --w has no effect. Use --w or --all-windows.")


if __name__ == "__main__":
    main()

