from typing import Dict, Any, List, Optional
import math
import pandas as pd
from collections import defaultdict
import argparse

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood import (
    _bias_key,
    _sort_bias_keys,
    order_algo_modules_paper,
    short_name_from_algo,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx

# ----- KONFIGURACE -----

GROUP_TYPES = ["similar", "outlier", "random", "divergent", "variance"]
EVAL_TYPE = "test"

WINDOW_SIZE=10
NDCG_K = 20           # parametrizace K

ALGOS = [
    "eval_hybrid_general_rec_individual.py",
    "eval_hybrid_updatable.py",
    "eval_async_static_policy_simple_priority_function_group_rec.py",
    "eval_async_static_policy_simple_priority_function_individual_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
    "eval_sync_without_feedback.py",
    "eval_sync_with_feedback_ema.py",
]

# ----- METRIKY -----

# Internal metric keys -> text-mode column titles (console)
DISPLAY_COLUMNS = [
    ("per_user_ndcg_mean_overall", "NDCG@{k}_mean"),
    ("per_user_ndcg_min_overall", "NDCG@{k}_min"),
    ("ndcg_com_mean_overall", "NDCG@{k}_com"),
    ("ndcg_com_min_overall", "NDCG@{k}_com_min"),
]


def latex_ndcg_data_column_headers(k: int) -> List[str]:
    """Exact thesis headers (multicolumn + siunitx-friendly)."""
    kk = str(k)
    return [
        rf"\multicolumn{{1}}{{c}}{{$\mathrm{{NDCG@{kk}}}_{{\mathrm{{mean}}}}$}}",
        rf"\multicolumn{{1}}{{c}}{{$\mathrm{{NDCG@{kk}}}_{{\mathrm{{min}}}}$}}",
        rf"\multicolumn{{1}}{{c}}{{$\mathrm{{NDCG@{kk}}}_{{\mathrm{{com}}}}$}}",
        rf"\multicolumn{{1}}{{c}}{{$\mathrm{{NDCG@{kk}}}_{{\mathrm{{com_{{min}}}}}}$}}",
    ]


def _pick_bias_key(
    group_data: Dict[Any, Any],
    bias_idx: int,
    population_bias: Optional[float],
) -> Optional[Any]:
    """
    Vyber klíč biasu ve větvi ``group_type`` — buď podle ``--population-bias`` (β v datech),
    nebo podle ``--bias-index`` vůči seřazeným klíčům (0 → nejnižší β, typicky 0.0).
    """
    if not isinstance(group_data, dict) or not group_data:
        return None
    keys_sorted = _sort_bias_keys(list(group_data.keys()))
    if population_bias is not None:
        target = float(population_bias)
        for k in keys_sorted:
            try:
                if _bias_key(k) == target:
                    return k
            except (TypeError, ValueError):
                continue
        return None
    if bias_idx < 0 or bias_idx >= len(keys_sorted):
        return None
    return keys_sorted[bias_idx]


# ----- DATOVÉ NAČÍTÁNÍ -----

def load_ndcg_metrics(
    algos: List[str],
    window_size: str,
    eval_type: str,
    ndcg_key: str,
    bias_idx: int,
    groups_count: Optional[int] = None,
    population_bias: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    algo2metrics: Dict[str, Dict[str, float]] = {}

    for algo_name in algos:
        try:
            loaded = load_eval_res(
                algo_name, window_size, eval_type, groups_count=groups_count
            )
        except Exception as e:
            print(f"⚠️  Chyba při načítání {algo_name}: {e}")
            continue

        temp_accumulator = defaultdict(list)

        for group_type in GROUP_TYPES:
            group_data = loaded.get(group_type, {})
            if not isinstance(group_data, dict):
                continue

            bias_key = _pick_bias_key(group_data, bias_idx, population_bias)
            if bias_key is None:
                continue

            bias_data = group_data[bias_key]

            ndcg_block = bias_data.get(ndcg_key)
            if not isinstance(ndcg_block, dict):
                continue

            for metric, _ in DISPLAY_COLUMNS:
                val = ndcg_block.get(metric, math.nan)
                if isinstance(val, (int, float)) and not math.isnan(val):
                    temp_accumulator[metric].append(val)

        algo2metrics[algo_name] = {
            metric: (sum(vals) / len(vals)) if vals else math.nan
            for metric, vals in temp_accumulator.items()
        }

    return algo2metrics

# ----- TABULKA -----

def build_ndcg_dataframe(data: Dict[str, Dict[str, float]], algos: List[str], k: int) -> pd.DataFrame:
    rows = []
    for algo_name in order_algo_modules_paper(algos):
        algo_metrics = data.get(algo_name, {})
        row = {"algorithm": short_name_from_algo(algo_name)}
        for metric, _ in DISPLAY_COLUMNS:
            row[metric] = algo_metrics.get(metric, math.nan)
        rows.append(row)

    metric_cols = [m for m, _ in DISPLAY_COLUMNS]
    df = pd.DataFrame(rows, columns=["algorithm"] + metric_cols)
    rename_map = {m: label.format(k=k) for m, label in DISPLAY_COLUMNS}
    return df.rename(columns=rename_map)


def create_ndcg_latex_table(
    df: pd.DataFrame,
    k: int,
    bias_idx: int,
    population_bias: Optional[float],
) -> str:
    metric_cols = [c for c in df.columns if c != "algorithm"]

    generator = LaTeXTableGeneratorSIUnitx(
        df,
        column_specs=[(1, 3)] * len(metric_cols),
        column_width=1.8,
    )

    if population_bias is not None:
        cap_bias = rf"population bias $\beta={population_bias}$"
        lbl = f"tab:ndcg_at_{k}_beta_{str(population_bias).replace('.', '_')}"
    else:
        cap_bias = rf"bias index {bias_idx} (po seřazení klíčů $\beta$)"
        lbl = f"tab:ndcg_at_{k}_bias_{bias_idx}"

    latex_code = generator.generate_table(
        caption=rf"Výsledky NDCG@{k} pro {cap_bias} a jednotlivé algoritmy.",
        label=lbl,
        cell_bold_fn=lambda row_idx, col_idx, val: (
            col_idx >= 1 and pd.notna(val) and val == df.iloc[:, col_idx].max(skipna=True)
        ),
        data_column_headers=latex_ndcg_data_column_headers(k),
    )
    return latex_code


def create_ndcg_text_table(df: pd.DataFrame) -> str:
    printable = df.copy()
    for c in printable.columns:
        if c == "algorithm":
            continue
        printable[c] = printable[c].map(
            lambda x: f"{x:.6f}" if isinstance(x, (int, float)) and not math.isnan(float(x)) else "nan"
        )
    return printable.to_string(index=False)

# ----- MAIN -----
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Top-k values to consider in NDCG@k")
    parser.add_argument("--window-size", type=int, default=10, help="Window size")
    parser.add_argument(
        "--groups-count",
        type=int,
        default=None,
        help="Match cache segment eval_n_<N> (same as eval --groups-count), e.g. 1000 for paper runs.",
    )
    parser.add_argument(
        "--bias-index",
        type=int,
        default=0,
        help="Index of beta in sorted bias keys (0 = lowest beta, usually 0.0). Ignored if --population-bias is set.",
    )
    parser.add_argument(
        "--population-bias",
        type=float,
        default=None,
        help="Explicit population beta in stored results (e.g. 0, 1, 2). Overrides --bias-index.",
    )
    parser.add_argument(
        "--output",
        choices=["latex", "text"],
        default="latex",
        help="latex: thesis table; text: plain table to stdout.",
    )
    args = parser.parse_args()

    NDCG_K = args.k
    WINDOW_SIZE = args.window_size
    NDCG_KEY = f"ndcg@{NDCG_K}"

    if args.population_bias is not None:
        print(
            f"Printing NDCG@{NDCG_K} window size {WINDOW_SIZE} "
            f"population_bias={args.population_bias} as {args.output}"
        )
    else:
        print(
            f"Printing NDCG@{NDCG_K} window size {WINDOW_SIZE} "
            f"bias_index={args.bias_index} as {args.output}"
        )

    ndcg_data = load_ndcg_metrics(
        ALGOS,
        WINDOW_SIZE,
        eval_type=EVAL_TYPE,
        ndcg_key=NDCG_KEY,
        bias_idx=args.bias_index,
        groups_count=args.groups_count,
        population_bias=args.population_bias,
    )
    df = build_ndcg_dataframe(ndcg_data, ALGOS, NDCG_K)
    if args.output == "text":
        print(create_ndcg_text_table(df))
    else:
        print(create_ndcg_latex_table(df, NDCG_K, args.bias_index, args.population_bias))
