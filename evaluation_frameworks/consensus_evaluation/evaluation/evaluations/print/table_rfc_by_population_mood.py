from collections import defaultdict
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd
import argparse

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.rfc_table_metric_spec import (
    add_rfc_metric_arg,
    resolve_rfc_metric,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx

# ----- konfigurace -----

GROUP_TYPES: List[str] = ["similar", "outlier", "random", "divergent", "variance"]

EVAL_TYPE: str = "test"

ALGOS: List[str] = [
    "eval_hybrid_general_rec_individual.py",
    "eval_hybrid_updatable.py",
    "eval_async_static_policy_simple_priority_function_group_rec.py",
    "eval_async_static_policy_simple_priority_function_individual_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
    "eval_sts_group_dynamic.py",
    #"eval_sync_with_feedback_ema.py",
    "eval_sync_without_feedback.py",
    "eval_sync_with_feedback_ema.py",
]

SHORT_NAME_BY_ALGO: Dict[str, str] = {
    "eval_async_static_policy_simple_priority_function_individual_rec.py": "Async-Static-Ind",
    "eval_async_static_policy_simple_priority_function_group_rec.py": "Async-Static-Grp",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py": "Async-Dyn-Ind",
    "eval_sts_group_dynamic.py": "STS-Dyn",
    "eval_hybrid_general_rec_individual.py": "Hybrid-Ind",
    "eval_hybrid_updatable.py": "Hybrid-Grp-EMA",
    "eval_sync_without_feedback.py": "Sync",
    "eval_sync_with_feedback_ema.py": "Sync-EMA",
}

# Pořadí řádků v tabulkách: hybrid → async → sync (v rámci rodiny stabilní pořadí jako v SHORT_NAME_BY_ALGO).
PAPER_ORDER_SHORT_NAMES: Tuple[str, ...] = (
    "Hybrid-Ind",
    "Hybrid-Grp-EMA",
    "Async-Static-Ind",
    "Async-Static-Grp",
    "Async-Dyn-Ind",
    "STS-Dyn",
    "Sync",
    "Sync-EMA",
)


def order_algo_modules_paper(algos: List[str]) -> List[str]:
    """Seřadí názvy eval modulů podle PAPER_ORDER_SHORT_NAMES; neznámé moduly na konec."""
    rank = {n: i for i, n in enumerate(PAPER_ORDER_SHORT_NAMES)}

    def sort_key(mod: str) -> tuple[int, str]:
        short = short_name_from_algo(mod)
        return (rank.get(short, 1000), short)

    return sorted(algos, key=sort_key)


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


def short_name_from_algo(algo_name: str) -> str:
    return SHORT_NAME_BY_ALGO.get(algo_name, algo_name.replace(".py", ""))


def display_name_for_slug(slug2algo: Dict[str, str], slug: str) -> str:
    algo_name = slug2algo.get(slug, "")
    return short_name_from_algo(algo_name)


def load_rfc_values(algos: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Vrátí data ve struktuře:
    Dict[algo_name, Dict[group_type, EvaluationDictData]]
    """
    algo2data: Dict[str, Dict[str, Any]] = {}
    for algo_name in algos:
        try:
            results = load_eval_res(algo_name, "test")
            algo2data[algo_name] = results
        except ValueError:
            print(f"WARNING: Evaluation results for the {algo_name} algorithm not found!!!")
            continue
    return algo2data

def _safe_get_metric(d: Dict[str, Any], metric_key: str = "average") -> Optional[float]:
    """
    Bezpečně vytáhne metrickou hodnotu (default 'average') z EvaluationDictData.
    Podporuje i variantu s vnořeným 'metrics' slovníkem.
    """
    if d is None:
        return math.nan
    # 1) přímý klíč
    if metric_key in d:
        try:
            return float(d[metric_key])
        except (TypeError, ValueError):
            pass
    # 2) vnořené 'metrics'
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
    val = _safe_get_metric(d, metric_key=metric_key)
    if pd.notna(val):
        return float(val)
    if metric_key != "first_consensus_global_position_across_groups":
        return val
    avg_round = _safe_get_metric(d, metric_key="average")
    rank = _safe_get_metric(d, metric_key="first_consensus_rank_across_groups")
    if pd.notna(avg_round) and pd.notna(rank):
        return float(window_size) * (float(avg_round) - 1.0) + float(rank)
    return math.nan


def _bias_key(b: Any) -> Any:
    """Normalize pickle keys (0 vs 0.0 vs \"0\") to float for stable columns."""
    if isinstance(b, bool):
        return b
    if isinstance(b, (int, float)):
        return float(b)
    if isinstance(b, str):
        try:
            return float(b)
        except ValueError:
            return b
    return b


def _sort_bias_keys(keys: List[Any]) -> List[Any]:
    def sort_key(b: Any):
        try:
            return (0, float(b))
        except (TypeError, ValueError):
            return (1, str(b))

    return sorted(keys, key=sort_key)


def create_table_bias_columns(
    results: Dict[str, Dict[str, Any]],
    algos: List[str],
    metric_key: str = "average",
    adjust_A: bool = True,
    bias_columns: Optional[List[Any]] = None,
    *,
    apply_async_minus_one: bool = True,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Vytvoří LaTeX tabulku:
        - řádky: algoritmy (slugy)
        - sloupce: bias klíče
        - hodnoty: průměrné metriky; pokud je v results dict, vezme se results[algo][bias][metric_key]

        Pokud ``apply_async_minus_one`` a ``adjust_A`` a slug obsahuje 'A':
            - název algoritmu dostane suffix (RFC_{adj.})
            - všechny hodnoty v řádku se sníží o 1 (jen u metriky „kola do shody“)
            - tyto řádky se ignorují při hledání minim (bold).
    """
    slug2algo = strategy_2_slug(algos)
    algo_to_slug = {algo: slug for slug, algo in slug2algo.items()}

    if bias_columns is not None:
        all_biases = [_bias_key(b) for b in bias_columns]
    else:
        all_biases = _sort_bias_keys(
            list(
                {
                    _bias_key(bias)
                    for algo in slug2algo.values()
                    if algo in results and isinstance(results[algo], dict)
                    for bias in results[algo].keys()
                }
            )
        )

    def coerce_to_float(v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            if metric_key in v and isinstance(v[metric_key], (int, float)):
                return float(v[metric_key])
            try:
                return float(_safe_get_metric(v, metric_key=metric_key))
            except Exception:
                pass
        return math.nan

    # postav řádky (pořadí jako v přehledové tabulce variant, ne A/H/S z technických slugů)
    rows = []
    for algo_name in order_algo_modules_paper(list(slug2algo.values())):
        slug = algo_to_slug[algo_name]
        is_adj = apply_async_minus_one and adjust_A and ("A" in slug)
        algo_label = short_name_from_algo(algo_name)
        row = {"algorithm": algo_label + (" (RFC$_{adj.}$)" if is_adj else "")}
        bias_values = results.get(algo_name, {}) or {}
        for bias in all_biases:
            raw_val = bias_values.get(bias, math.nan)
            if isinstance(raw_val, float) and math.isnan(raw_val):
                for k in bias_values:
                    if _bias_key(k) == bias:
                        raw_val = bias_values[k]
                        break
            val = coerce_to_float(raw_val)
            if is_adj and pd.notna(val):
                val = val - 1
            row[bias] = val
        rows.append(row)

    # DataFrame v pořadí: algorithm + bias sloupce
    df = pd.DataFrame(rows, columns=["algorithm"] + all_biases)

    # připrav column_specs pro všechny numerické sloupce
    column_specs = [(1, 2) for _ in all_biases]

    generator = LaTeXTableGeneratorSIUnitx(
        df,
        column_specs=column_specs,
        column_width=1.5 if len(all_biases) <= 5 else 1.2,
    )

    cap = caption or rf"Porovnání algoritmů skrze biasy (metrika: {metric_key})."
    lbl = label or "tab:algo_bias_comparison"
    # bold jen minima mezi ne-adj řádky
    latex_code = generator.generate_table(
        caption=cap,
        label=lbl,
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

    # df of slug–algoritmus pairs (řádky v pořadí přehledové tabulky)
    df = pd.DataFrame(
        [
            {
                "slug": algo_to_slug[algo],
                "short_name": short_name_from_algo(algo),
                "algorithm": algo,
            }
            for algo in order_algo_modules_paper(list(slug2algo.values()))
        ],
        columns=["slug", "short_name", "algorithm"],
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

def _load_single_algo_per_group_biases(
    algo_name: str,
    window_size: str,
    *,
    metric_key: str = "average",
    eval_type: str = EVAL_TYPE,
    groups_count: Optional[int] = None,
    merge_all_pickles: bool = True,
    group_types: Optional[List[str]] = None,
) -> Dict[str, Dict[Any, float]]:
    """
    Pro jeden algoritmus: ``{group_type: {bias_key: metric}}`` (bez průměrování napříč skupinami).
    """
    gts = group_types if group_types is not None else GROUP_TYPES
    per_group_bias_vals: Dict[str, Dict[Any, float]] = {g: {} for g in gts}
    try:
        w_int = int(window_size)
    except (TypeError, ValueError):
        w_int = 10

    loaded = None
    try:
        loaded = load_eval_res(
            algo_name,
            window_size,
            eval_type,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
        )
    except (ValueError, OSError) as e:
        print(f"WARNING: Loading results for {algo_name} failed ({e}). Using NaNs.")
        return per_group_bias_vals

    if not isinstance(loaded, dict):
        return per_group_bias_vals

    for g_type in gts:
        group_dict = loaded.get(g_type)
        if not isinstance(group_dict, dict):
            if group_dict is None:
                print(f"WARNING: Group '{g_type}' missing for {algo_name}. Using NaNs.")
            else:
                print(f"WARNING: Group '{g_type}' for {algo_name} is not a dict. Using NaNs.")
            continue

        for users_bias, eval_dict in group_dict.items():
            val = _metric_with_fallback(
                eval_dict,
                metric_key=metric_key,
                window_size=w_int,
            )
            per_group_bias_vals[g_type][_bias_key(users_bias)] = val

    return per_group_bias_vals


def load_rfc_values_streaming(
    algos: List[str],
    window_size: str,
    metric_key: str = "average",
    eval_type: str = EVAL_TYPE,
    groups_count: Optional[int] = None,
    *,
    merge_all_pickles: bool = True,
    group_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Paměťově úsporné načítání: pro každý algoritmus zvlášť načti pickle,
    hned z něj vytáhni jen potřebnou metriku pro každou skupinu a zbytek zahoď.
    Když se něco nepodaří načíst, vypíše WARNING a vrátí NaNy.
    Vrací strukturu kompatibilní s create_table:
      {algo_name: {bias: average_value_across_groups}}

    ``merge_all_pickles`` (výchozí True): předá se do ``load_eval_res`` — sloučí všechna ``N.pkl``
    v kandidátních složkách, ne jen nejnovější podle mtime (jinak často chybí část biasů → NaN).
    """
    algo2data: Dict[str, Dict[str, Any]] = {}
    gts = group_types if group_types is not None else GROUP_TYPES

    for algo_name in algos:
        per_group_bias_vals = _load_single_algo_per_group_biases(
            algo_name,
            window_size,
            metric_key=metric_key,
            eval_type=eval_type,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
            group_types=gts,
        )

        agg: Dict[Any, List[float]] = defaultdict(list)
        for g_type, bias_dict in per_group_bias_vals.items():
            for bias, val in bias_dict.items():
                if not math.isnan(val):
                    agg[_bias_key(bias)].append(val)

        algo2data[algo_name] = {bias: (sum(vals) / len(vals)) for bias, vals in agg.items()}

    return algo2data


def load_rfc_values_by_group_streaming(
    algos: List[str],
    window_size: str,
    metric_key: str = "average",
    eval_type: str = EVAL_TYPE,
    groups_count: Optional[int] = None,
    *,
    merge_all_pickles: bool = True,
    group_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[Any, float]]]:
    """
    Stejné načtení jako streaming, ale bez průměrování napříč ``group_type``:
    ``{algo_name: {group_type: {bias: metric}}}``.
    """
    gts = group_types if group_types is not None else GROUP_TYPES
    out: Dict[str, Dict[str, Dict[Any, float]]] = {}
    for algo_name in algos:
        out[algo_name] = _load_single_algo_per_group_biases(
            algo_name,
            window_size,
            metric_key=metric_key,
            eval_type=eval_type,
            groups_count=groups_count,
            merge_all_pickles=merge_all_pickles,
            group_types=gts,
        )
    return out

# ----- main -----

def _latex_safe_label_suffix(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", default=10, help="Window size")
    parser.add_argument(
        "--biases",
        nargs="*",
        type=float,
        default=None,
        metavar="B",
        help="Explicitní sloupce biasů (pořadí v tabulce). Chybějící = prázdné/NaN. "
        "Výchozí: sjednocení klíčů z cache.",
    )
    parser.add_argument(
        "--by-group-type",
        action="store_true",
        help="Místo jedné tabulky (průměr přes typy skupin similar/outlier/…) vytiskni zvlášť "
        "tabulku pro každý group_type — řádky algoritmy, sloupce stále biasy.",
    )
    parser.add_argument(
        "--group-types",
        nargs="*",
        default=None,
        metavar="GT",
        help="Podmnožina typů skupin (pořadí výpisu). Výchozí: stejné jako v eval "
        f"({', '.join(GROUP_TYPES)}).",
    )
    parser.add_argument(
        "--latest-pickle-only",
        action="store_true",
        help="Načíst jen jeden nejnovější .pkl (staré chování). Výchozí je sloučit všechna N.pkl "
        "ve všech kandidátních složkách, aby se doplnily chybějící biasy z různých běhů.",
    )
    add_rfc_metric_arg(parser)
    args = parser.parse_args()

    spec = resolve_rfc_metric(args.rfc_metric)
    w_key = str(args.window_size)
    group_types_eff: List[str] = (
        list(args.group_types) if args.group_types is not None else list(GROUP_TYPES)
    )

    mode = "by group_type × bias" if args.by_group_type else "aggregated over group types"
    print(
        f"Printing RFC by population biases ({mode}) for window size {w_key} "
        f"(metric={args.rfc_metric} → `{spec.storage_key}`)"
    )

    merge = not args.latest_pickle_only
    bias_columns = list(args.biases) if args.biases else None

    if args.by_group_type:
        nested = load_rfc_values_by_group_streaming(
            ALGOS,
            w_key,
            metric_key=spec.storage_key,
            eval_type=EVAL_TYPE,
            merge_all_pickles=merge,
            group_types=group_types_eff,
        )
        union = _sort_bias_keys(
            list(
                {
                    _bias_key(b)
                    for per_gt in nested.values()
                    if isinstance(per_gt, dict)
                    for bias_dict in per_gt.values()
                    if isinstance(bias_dict, dict)
                    for b in bias_dict
                }
            )
        )
    else:
        evaluation_data = load_rfc_values_streaming(
            ALGOS,
            w_key,
            metric_key=spec.storage_key,
            eval_type=EVAL_TYPE,
            merge_all_pickles=merge,
            group_types=group_types_eff,
        )
        union = _sort_bias_keys(
            list({_bias_key(b) for d in evaluation_data.values() if isinstance(d, dict) for b in d})
        )

    print(f"[table_rfc_by_population_mood] bias keys in cache (union over algos): {union}")
    if len(union) <= 1:
        print(
            "[table_rfc_by_population_mood] ⚠ jen jeden bias v pickle — typicky buď běžel jen jeden "
            "--population-biases, nebo starší upsert přepsal celý group_type větev. "
            "Od teď upsert slučuje biasy; pro doplnění znovu spusť eval se všemi biasy, nebo použij --biases.",
            flush=True,
        )

    print(create_slug_table(ALGOS))
    print()

    if args.by_group_type:
        for gt in group_types_eff:
            evaluation_data_gt: Dict[str, Dict[str, Any]] = {
                algo: (nested.get(algo) or {}).get(gt, {}) or {} for algo in ALGOS
            }
            slug = _latex_safe_label_suffix(gt)
            table_latex = create_table_bias_columns(
                evaluation_data_gt,
                ALGOS,
                metric_key=spec.storage_key,
                apply_async_minus_one=spec.subtract_one_for_async_slug,
                bias_columns=bias_columns,
                caption=(
                    rf"Porovnání algoritmů skrze biasy — typ skupiny \texttt{{{gt}}} "
                    rf"({spec.latex_caption_cs})."
                ),
                label=f"tab:algo_bias_comparison_{slug}",
            )
            print(table_latex)
            print()
    else:
        table_latex = create_table_bias_columns(
            evaluation_data,
            ALGOS,
            metric_key=spec.storage_key,
            apply_async_minus_one=spec.subtract_one_for_async_slug,
            bias_columns=bias_columns,
            caption=rf"Porovnání algoritmů skrze biasy ({spec.latex_caption_cs}).",
        )
        print(table_latex)