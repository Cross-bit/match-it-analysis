"""
Shared base for ``eval_*.py`` and ``tune_*.py`` experiment entrypoints.

* ``ConsensusExperimentBase`` — declares defaults (window ``W``, group types, biases, NDCG cutoffs) and
  hooks ``compute_results()`` / ``make_table()``; CLI is built from ``build_autorun_argparser()``.
* Companion helpers — RFC/LaTeX table builders and bias picking for multi-bias runs.

Persistence and ``--mode`` live in ``config.autorun`` (pickle layout under ``cache/cons_evaluations/``).
"""

from __future__ import annotations

import argparse
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, TypeVar

C = TypeVar("C", bound="ConsensusExperimentBase")


def build_autorun_argparser(description: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description or "Run or load evaluation results (standardized autorunner)."
    )
    p.add_argument(
        "--mode",
        choices=["auto", "compute", "load"],
        default="auto",
        help="auto: load if available, else compute; compute: always compute; load: load only.",
    )
    p.add_argument(
        "--num",
        type=int,
        default=None,
        help="Specific cached run number (e.g. 5 → loads '5.pkl'). If omitted, the latest run is used.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override default consensus window size W (--window-size in cache path w_<W>).",
    )
    p.add_argument(
        "--groups-count",
        "--group-eval-size",
        type=int,
        default=None,
        dest="groups_count",
        metavar="N",
        help="How many groups to run through the evaluator (Runner evaluation_size). "
        "Alias: --group-eval-size (old name).",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Override default group cardinality for large-group evals (users per group).",
    )
    p.add_argument(
        "--population-biases",
        nargs="+",
        type=float,
        default=None,
        metavar="B",
        help="Population mood biases to simulate (default: experiment's DEFAULT_POPULATION_BIASES). "
        "Example: fast run with only unbiased mood: --population-biases 0",
    )
    p.add_argument(
        "--ndcg-k",
        nargs="+",
        type=int,
        default=None,
        dest="ndcg_ks",
        metavar="K",
        help="NDCG cutoffs K to compute (default: experiment's DEFAULT_NDCG_KS). Example: --ndcg-k 20",
    )
    p.add_argument(
        "--group-types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="Subset of group types to evaluate (default: all). Example: --group-types similar outlier divergent",
    )
    p.add_argument(
        "--fast-grid",
        action="store_true",
        help="Tune-only: coarser sigmoid/hybrid hyperparameter grid (fewer Runner.run; see tune_* FAST_SIGMOID_PARAMS).",
    )
    p.add_argument(
        "--save-mode",
        choices=["upsert", "append"],
        default="upsert",
        help="How to persist computed results: upsert updates latest pickle by group type, append writes a new numbered pickle.",
    )
    p.add_argument(
        "--debug-profile",
        action="store_true",
        help="Write per-stage timing profile to cache/cons_evaluations/_debug_profiles/*.jsonl.",
    )
    p.add_argument(
        "--debug-profile-tag",
        default=None,
        help="Optional tag appended to debug profile filename (e.g. workers4).",
    )
    return p


def pick_bias_result_for_table(bias_to_res: Dict[Any, Any]) -> Any:
    """
    ``Runner.run`` ukládá ``{population_mood_bias: stats_dict}`` (klíč je float, ne index 0).
    ``make_table`` potřebuje jeden blok metrik: jediný bias z běhu, nebo preferovaně 0 při více biasích.
    """
    if not isinstance(bias_to_res, dict) or not bias_to_res:
        raise KeyError("pick_bias_result_for_table: prázdná nebo neplatná mapa bias → výsledky")
    if len(bias_to_res) == 1:
        return next(iter(bias_to_res.values()))
    for k in (0, 0.0):
        if k in bias_to_res:
            return bias_to_res[k]
    numeric_keys = [x for x in bias_to_res if type(x) in (int, float)]
    if numeric_keys:
        k0 = min(numeric_keys, key=float)
        return bias_to_res[k0]
    return next(iter(bias_to_res.values()))


def tune_group_types_present(results: Dict[str, Any], preferred_order: Sequence[str]) -> List[str]:
    """
    Pro ``tune_*`` výsledky ``{group_type -> ...}`` vrátí jen ty ``group_type``, které v pickle opravdu jsou.

    Starší / částečné běhy často nemají ``divergent`` / ``variance`` — pak původní ``make_table`` padal na ``KeyError``.
    """
    return [str(gt) for gt in preferred_order if gt in results]


def rfc_metric_from_picked_stats(picked: Any, *, metric_key: str = "average") -> float:
    """Vytáhni skalární metriku z bloku po ``pick_bias_result_for_table`` (``average`` nahoře nebo pod ``metrics``)."""
    import math

    if not isinstance(picked, dict):
        return float("nan")
    if metric_key in picked:
        try:
            v = float(picked[metric_key])
            return v if math.isfinite(v) else float("nan")
        except (TypeError, ValueError):
            pass
    metrics = picked.get("metrics")
    if isinstance(metrics, dict) and metric_key in metrics:
        try:
            v = float(metrics[metric_key])
            return v if math.isfinite(v) else float("nan")
        except (TypeError, ValueError):
            pass
    return float("nan")


def rfc_average_from_tune_result_cell(res_cell: Any) -> float:
    """
    Jedna buňka z ``tune_*`` tabulky: ``Runner.run`` → ``{bias -> stats}`` → RFC ``average``.
    """
    if res_cell is None or not isinstance(res_cell, dict) or not res_cell:
        return float("nan")
    try:
        picked = pick_bias_result_for_table(res_cell)
    except KeyError:
        return float("nan")
    return rfc_metric_from_picked_stats(picked, metric_key="average")


def latex_rfc_table_group_types_by_biases(
    *,
    results: Dict[str, Dict[Any, Any]],
    group_types: List[str],
    metric_key: str = "average",
    caption: str,
    label: str,
) -> str:
    """
    Kompletní RFC tabulka pro jeden experiment: řádky = ``group_types``, sloupce = všechny
    ``population_mood_bias`` klíče nalezené v ``results[group_type]``.
    """
    import math

    import pandas as pd

    from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx

    bias_keys: set = set()
    for gt in group_types:
        block = results.get(gt)
        if isinstance(block, dict):
            bias_keys.update(block.keys())

    def _sort_bias(b: Any):
        try:
            return (0, float(b))
        except (TypeError, ValueError):
            return (1, str(b))

    all_biases = sorted(bias_keys, key=_sort_bias)
    if not all_biases:
        return (
            f"% {label}: žádné bias klíče v results (prázdná tabulka)\n"
            f"% \\caption{{{caption}}}\n"
        )

    str_cols = [str(b) for b in all_biases]
    rows = []
    for gt in group_types:
        block = results.get(gt) if isinstance(results.get(gt), dict) else {}
        row: Dict[str, Any] = {"group_type": gt}
        row_vals: List[float] = []
        for b, sc in zip(all_biases, str_cols):
            st = block.get(b) if isinstance(block, dict) else None
            if isinstance(st, dict) and metric_key in st:
                try:
                    v = float(st[metric_key])
                except (TypeError, ValueError):
                    v = float("nan")
            else:
                v = float("nan")
            row[sc] = v
            if not math.isnan(v):
                row_vals.append(v)
        row["mean"] = sum(row_vals) / len(row_vals) if row_vals else float("nan")
        rows.append(row)

    columns = ["group_type"] + str_cols + ["mean"]
    df = pd.DataFrame(rows, columns=columns)
    column_specs = [(1, 2)] * (len(columns) - 1)
    gen = LaTeXTableGeneratorSIUnitx(df, column_specs=column_specs, column_width=1.35)
    return gen.generate_table(
        caption=caption,
        label=label,
        cell_bold_fn=lambda ri, ci, val: (
            ci >= 1
            and pd.notna(val)
            and ci < len(columns) - 1
            and val == df.iloc[:, ci].min(skipna=True)
        ),
    )


class ConsensusExperimentBase(ABC):
    """
    Base for eval / tune entrypoints: hyperparameters live on the instance.
    Subclasses implement compute_results() and make_table(); optional stats(results) -> str.
    """

    DEFAULT_W_SIZE: int = 10
    DEFAULT_GROUPS_COUNT: int = 1000
    DEFAULT_EVAL_TYPE: Literal["train", "validation", "test"] = "validation"
    DEFAULT_GROUP_SIZE: Optional[int] = None

    def __init__(
        self,
        *,
        evaluation_name: str,
        eval_type: Literal["train", "validation", "test"],
        w_size: int,
        groups_count: int,
        group_size: Optional[int] = None,
    ):
        self.evaluation_name = evaluation_name
        self.eval_type = eval_type
        self.w_size = w_size
        self.groups_count = groups_count
        self.group_size = group_size

    @classmethod
    def evaluation_name_for_class(cls) -> str:
        return Path(inspect.getfile(cls)).name

    @classmethod
    def core_from_cli(cls: Type[C], args: argparse.Namespace) -> Dict[str, Any]:
        w = args.window_size if args.window_size is not None else cls.DEFAULT_W_SIZE
        gc = args.groups_count if args.groups_count is not None else cls.DEFAULT_GROUPS_COUNT
        gs = args.group_size if args.group_size is not None else cls.DEFAULT_GROUP_SIZE

        if args.window_size is not None:
            print(f"[config] Overriding W_SIZE: {cls.DEFAULT_W_SIZE} → {w}")
        if args.groups_count is not None:
            print(f"[config] Overriding GROUPS_COUNT: {cls.DEFAULT_GROUPS_COUNT} → {gc}")
        if args.group_size is not None:
            print(f"[config] Overriding GROUP_SIZE (cardinality): {cls.DEFAULT_GROUP_SIZE} → {gs}")

        return {
            "evaluation_name": cls.evaluation_name_for_class(),
            "eval_type": cls.DEFAULT_EVAL_TYPE,
            "w_size": w,
            "groups_count": gc,
            "group_size": gs,
        }

    @classmethod
    def _apply_optional_cli_overrides(cls, args: argparse.Namespace, kw: Dict[str, Any]) -> None:
        """If CLI passed metric/group overrides and this experiment's __init__ accepts them, merge into kw."""
        sig = inspect.signature(cls.__init__)
        names = sig.parameters
        if "population_biases" in names and getattr(args, "population_biases", None) is not None:
            kw["population_biases"] = list(args.population_biases)
            print(f"[config] population_biases override → {kw['population_biases']}")
        if "ndcg_ks" in names and getattr(args, "ndcg_ks", None) is not None:
            kw["ndcg_ks"] = list(args.ndcg_ks)
            print(f"[config] ndcg_ks override → {kw['ndcg_ks']}")
        if "group_types" in names and getattr(args, "group_types", None) is not None:
            kw["group_types"] = list(args.group_types)
            print(f"[config] group_types override → {kw['group_types']}")
        if "fast_grid" in names and getattr(args, "fast_grid", False):
            kw["fast_grid"] = True
            print("[config] fast_grid=True → coarser hyperparameter grid")

    @classmethod
    def from_cli_args(cls: Type[C], args: argparse.Namespace) -> C:
        kw = cls.core_from_cli(args)
        cls._apply_optional_cli_overrides(args, kw)
        return cls(**kw)

    def cons_eval_set_progress_slot(self, slot_index_1based: int, *, n_slots: Optional[int] = None) -> None:
        """
        Nastaví env pro ``batch_run_progress`` před ``Runner.run``.
        ``n_slots`` = počet slotů v tomto modulu (výchozí: len(group_types), pokud existuje).
        """
        from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.batch_run_progress import (
            set_progress_slot_before_runner,
        )

        gts = getattr(self, "group_types", None)
        ns = n_slots if n_slots is not None else (len(gts) if gts is not None else 1)
        biases = getattr(self, "population_biases", None) or [0.0]
        set_progress_slot_before_runner(slot_index_1based, max(1, ns), max(1, len(biases)))

    @abstractmethod
    def compute_results(self) -> Any:
        ...

    @abstractmethod
    def make_table(self, results: Any) -> str:
        ...
