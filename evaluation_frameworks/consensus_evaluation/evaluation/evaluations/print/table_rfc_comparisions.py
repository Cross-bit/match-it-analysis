from typing import Any, Dict, List, Optional

import math
import pandas as pd
import argparse

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.rfc_table_metric_spec import (
    add_rfc_metric_arg,
    resolve_rfc_metric,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood import (
    order_algo_modules_paper,
    short_name_from_algo,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx

# ----- konfigurace -----

DEFAULT_GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]

EVAL_TYPE: str = "test"

ALGOS: List[str] = [
    #"eval_async_static_policy_simple_function_group_rec.py",
    "eval_hybrid_general_rec_individual.py",
    "eval_hybrid_updatable.py",
    "eval_async_static_policy_simple_priority_function_group_rec.py",
    "eval_async_static_policy_simple_priority_function_individual_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
    #"eval_sync_with_feedback_ema.py",
    "eval_sync_without_feedback.py",
    "eval_sync_with_feedback_ema.py",
]

# ----- pomocné funkce -----

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

def _safe_get_metric(d: Dict[str, Any], metric_key: str = "average") -> Optional[float]:
    """
    Bezpečně vytáhne metrickou hodnotu (default 'average') z EvaluationDictData.
    Podporuje i variantu s vnořeným 'metrics' slovníkem.
    """
    if d is None:
        return math.nan
    # 1) direct key
    if metric_key in d:
        try:
            return float(d[metric_key])
        except (TypeError, ValueError):
            pass
    # 2) nested 'metrics'
    metrics = d.get("metrics") if isinstance(d, dict) else None
    if isinstance(metrics, dict) and (metric_key in metrics):
        try:
            return float(metrics[metric_key])
        except (TypeError, ValueError):
            pass
    return math.nan


def _metric_with_fallback(
    d: Dict[str, Any],
    *,
    metric_key: str,
    window_size: int,
) -> float:
    """
    Read metric from stats block; for cards_seen_until_consensus support backfill
    from older caches that only have `average` and `first_consensus_rank_across_groups`.
    """
    val = _safe_get_metric(d, metric_key=metric_key)
    if pd.notna(val):
        return float(val)
    if metric_key != "first_consensus_global_position_across_groups":
        return val

    avg_round = _safe_get_metric(d, metric_key="average")
    rank = _safe_get_metric(d, metric_key="first_consensus_rank_across_groups")
    if pd.notna(avg_round) and pd.notna(rank):
        # E[(r-1)*w + p] = w*(E[r]-1) + E[p]
        return float(window_size) * (float(avg_round) - 1.0) + float(rank)
    return math.nan


def create_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    algos: List[str],
    group_types: List[str],
    metric_key: str = "average",
    adjust_A: bool = False,
    *,
    caption_metric_label: Optional[str] = None,
) -> str:
    """
    Sestaví LaTeX tabulku, kde:
        - řádky jsou algoritmy (slugs),
        - sloupce jsou group typy (včetně divergent) + průměr 'average'.
        - pokud `adjust_A=True` a slug obsahuje 'A' a metrika je „kola do shody“,
            přidá se suffix (RFC_{adj.}) a hodnoty se sníží o 1.
    """
    slug2algo = strategy_2_slug(algos)
    algo_to_slug = {algo: slug for slug, algo in slug2algo.items()}

    rows = []
    for algo_name in order_algo_modules_paper(list(slug2algo.values())):
        slug = algo_to_slug[algo_name]
        is_adj = adjust_A and ("A" in slug) and (metric_key == "average")
        base_label = short_name_from_algo(algo_name)
        row = {"algorithm": base_label + (" (RFC$_{adj.}$)" if is_adj else "")}

        for gt in group_types:
            val = math.nan
            if algo_name in results:
                group_dict = results[algo_name].get(gt)
                val = _safe_get_metric(group_dict, metric_key=metric_key)
            if is_adj and pd.notna(val):
                val = val - 1  # odečti 1 (jen u průměrného kola shody)
            row[gt] = val
        rows.append(row)

    # DataFrame v požadovaném pořadí
    df = pd.DataFrame(rows, columns=["algorithm"] + group_types)

    df["average"] = df[group_types].mean(axis=1, skipna=True)

    generator = LaTeXTableGeneratorSIUnitx(
        df,
        column_specs=None,
        column_width=1.5,
    )

    cap_lbl = caption_metric_label or ("RFC$_{adj.}$" if adjust_A and metric_key == "average" else "RFC")

    latex_code = generator.generate_table(
        caption=rf"Porovnání algoritmů — {cap_lbl}",
        label="tab:async_sync_comparision",
        cell_bold_fn=lambda row_idx, col_idx, val: (
            col_idx >= 1
            and pd.notna(val)
            and "adj." not in str(df.iloc[row_idx, 0])
            and val == df.loc[~df.iloc[:, 0].str.contains("adj.", regex=False), df.columns[col_idx]].min(skipna=True)
        ),
    )

    return latex_code


def create_slug_table(algos: List[str]) -> str:
    """
    Creates LATEX table of algorithm slugs -- S0, S1, A1, H0 ... to their full name
    """
    slug2algo = strategy_2_slug(algos)
    algo_to_slug = {algo: slug for slug, algo in slug2algo.items()}

    df = pd.DataFrame(
        [
            {"slug": algo_to_slug[algo], "algorithm": algo}
            for algo in order_algo_modules_paper(list(slug2algo.values()))
        ],
        columns=["slug", "algorithm"],
    )

    # Simple table generator
    generator = LaTeXTableGeneratorSIUnitx(
        df,
        column_specs=None,
        column_width=3.0,
    )

    latex_code = generator.generate_table(
        caption=r"Mapování zkratek (slugs) na algoritmy.",
        label="tab:slug_mapping",
    )
    return latex_code

def load_rfc_values_streaming(
    algos: List[str],
    window_size: str,
    metric_key: str = "average",
    eval_type: str = EVAL_TYPE,
) -> Dict[str, Dict[str, Any]]:
    """
    Memory-efficient loading: for each algorithm, load the pickle file
    separately, immediately extract only the required metric for each
    group, and discard the rest.

    Returns a structure compatible with `create_table`:
    {algo_name: {group_type: {"average": <float>}}}
    """

    algo2data: Dict[str, Dict[str, Any]] = {}
    try:
        w_int = int(window_size)
    except (TypeError, ValueError):
        w_int = 10
    for algo_name in algos:
        try:
            loaded = load_eval_res(algo_name, window_size, eval_type)
        except ValueError:
            print(f"WARNING: Evaluation results for the {algo_name} algorithm not found!!!")
            continue
        except OSError as e:
            print(f"WARNING: Loading results for {algo_name} failed ({e}). Using NaNs.")
            loaded = None

        # for given algo prepare mini-structure
        algo2data[algo_name] = {}
        present_group_types = [
            k for k in (loaded.keys() if isinstance(loaded, dict) else []) if isinstance(k, str)
        ] or list(DEFAULT_GROUP_TYPES)
        for g_type in present_group_types:
            val = math.nan
            if isinstance(loaded, dict):
                group_dict = loaded.get(g_type, {})
                if isinstance(group_dict, dict) and 0 in group_dict:
                    val = _metric_with_fallback(
                        group_dict[0],
                        metric_key=metric_key,
                        window_size=w_int,
                    )
                else:
                    val = math.nan

            # safe only miniature
            algo2data[algo_name][g_type] = {metric_key: val}

        # free
        loaded = None
    return algo2data


def detect_group_types(results: Dict[str, Dict[str, Any]], fallback: List[str]) -> List[str]:
    detected = set()
    for _, algo_block in results.items():
        if not isinstance(algo_block, dict):
            continue
        for key in algo_block.keys():
            if isinstance(key, str):
                detected.add(key)
    if not detected:
        return list(fallback)
    preferred = [g for g in fallback if g in detected]
    rest = sorted(g for g in detected if g not in preferred)
    return preferred + rest

# ----- main -----

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", default=10, help="Window size")
    add_rfc_metric_arg(parser)
    args = parser.parse_args()

    spec = resolve_rfc_metric(args.rfc_metric)
    print(
        f"Printing RFC comparison table for window size {args.window_size} "
        f"(metric={args.rfc_metric} → cache key `{spec.storage_key}`)"
    )

    evaluation_data = load_rfc_values_streaming(
        ALGOS, args.window_size, metric_key=spec.storage_key, eval_type=EVAL_TYPE
    )
    group_types = detect_group_types(evaluation_data, DEFAULT_GROUP_TYPES)

    # 2) print slugs
    print(create_slug_table(ALGOS))
    print()

    # 3) generate latex table
    table_latex = create_table(
        evaluation_data,
        ALGOS,
        group_types=group_types,
        metric_key=spec.storage_key,
        adjust_A=spec.subtract_one_for_async_slug,
        caption_metric_label=spec.latex_caption_short,
    )

    # 4) print LaTeX
    print(table_latex)