"""
Evaluation I/O and ``autorun()`` orchestration for all experiment modules.

**Role:** resolve cache paths (labeled vs legacy layout), run ``compute_results()`` in ``compute`` or
``load`` mode, optionally start JSONL timing via ``debug_profile`` when ``--debug-profile`` or
``CONS_EVAL_DEBUG_PROFILE=1``.

**Main entrypoints:** ``autorun()``, ``safe_eval_res`` / ``load_eval_res`` — used by every
``if __name__ == "__main__"`` block in ``eval_*`` and ``tune_*``.

Kořen cache výsledků: pokud existuje ``<algos-eval>/cache/cons_evaluations``, použije se ten
(jinak ``utils.config.CACHE_FILES_DIR/cons_evaluations``, typicky ``…/analysis/cache/``).

**Aktuální** cesta (složky mají v názvu parametr, aby šly číst v průzkumníku):

::

    cache/cons_evaluations/w_<W_fixed>/[group_<group_size>/]split_<split>/<název_modulu>.py/<N>.pkl

Segment ``group_<n>`` je jen když ``experiment.group_size`` není None.

``load_eval_res`` zkouší nejdřív tuto strukturu, pak **legacy** (starší běhy):

::

    cache/cons_evaluations/<W_fixed>/[large/<group_size>/]<split>/<název_modulu>.py/<N>.pkl

Úplný přehled: ``RESULT_CACHE_LAYOUT.md`` v tomto adresáři.
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Type

from utils.config import CACHE_FILES_DIR, load_from_pickle, load_or_build_pickle, save_to_pickle

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.base_experiment import (
    ConsensusExperimentBase,
    build_autorun_argparser,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.debug_profile import (
    end_session,
    get_profile_file,
    log_event,
    start_session,
    timed,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Dva běžné kořeny: (A) ``<text2>/analysis/cache/cons_evaluations`` z ``utils.config``,
# (B) ``<algos-eval>/cache/cons_evaluations`` (např. ``tree cache/`` z kořene tohoto repa).
# Když existuje (B), použij ho — jinak se tabulky/loader dívají jinam než reálná data na disku.
_REPO_CONS_EVALUATIONS = Path(__file__).resolve().parents[4] / "cache" / "cons_evaluations"
EVALUATIONS_DIR = (
    _REPO_CONS_EVALUATIONS
    if _REPO_CONS_EVALUATIONS.is_dir()
    else CACHE_FILES_DIR / "cons_evaluations"
)

LayoutKind = Literal["labeled", "legacy"]


def evaluation_results_dir(
    *,
    window_size: str,
    eval_type: Literal["train", "validation", "test"],
    evaluation_name: str,
    group_size: Optional[int] = None,
    groups_count: Optional[int] = None,
    group_types: Optional[Iterable[str]] = None,
    layout: LayoutKind = "labeled",
) -> Path:
    """
    Adresář s ``*.pkl`` pro danou kombinaci parametrů (bez čísla souboru).

    - ``labeled`` (default):
      ``w_<window>/[group_<n>/]split_<split>/[eval_n_<groups_count>/]<evaluation_name>``.
      Segment ``eval_n_*`` je volitelný (zpětně kompatibilní).
    - ``legacy``: starší tvar ``<window>/[large/<n>/]<split>/<evaluation_name>`` — jen pro čtení staré cache.
    """
    if layout == "labeled":
        base = EVALUATIONS_DIR / f"w_{window_size}"
        if group_size is not None:
            base = base / f"group_{group_size}"
        base = base / f"split_{eval_type}"
        if groups_count is not None:
            base = base / f"eval_n_{groups_count}"
        # group_types is intentionally NOT part of directory layout.
        # It serves as runtime metadata, not path cardinality.
        return base / evaluation_name

    base = EVALUATIONS_DIR / str(window_size)
    if group_size is not None:
        base = base / "large" / str(group_size)
    return base / eval_type / evaluation_name


def _load_eval_pickle_from_dir(evaluation_dir: Path, num: Optional[int]):
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Directory {evaluation_dir} does not exist.")

    if num is not None:
        file_path = evaluation_dir / f"{num}.pkl"
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        return load_from_pickle(file_path)

    existing = []
    for f in evaluation_dir.glob("*.pkl"):
        try:
            existing.append((int(f.stem), f))
        except ValueError:
            pass

    if not existing:
        raise FileNotFoundError(f"No numbered pickle files found in {evaluation_dir}.")

    _, file_path = max(existing, key=lambda x: x[0])
    return load_from_pickle(file_path)


def _list_numbered_pickles(evaluation_dir: Path):
    numbered = []
    if not evaluation_dir.is_dir():
        return numbered
    for f in evaluation_dir.glob("*.pkl"):
        try:
            numbered.append((int(f.stem), f))
        except ValueError:
            pass
    return numbered


def _load_eval_pickle_from_any_dir(dirs: List[Path], num: Optional[int]):
    """
    Try a list of candidate directories. If num is None, pick the most recently modified
    numbered pickle across all candidates.
    """
    existing_dirs = [d for d in dirs if d.is_dir()]
    if not existing_dirs:
        raise FileNotFoundError("No candidate cache directories exist.")

    if num is not None:
        errors = []
        for d in existing_dirs:
            try:
                return _load_eval_pickle_from_dir(d, num)
            except FileNotFoundError as e:
                errors.append(str(e))
        raise FileNotFoundError(" | ".join(errors))

    all_pickles = []
    for d in existing_dirs:
        for _, p in _list_numbered_pickles(d):
            all_pickles.append(p)

    if not all_pickles:
        raise FileNotFoundError(f"No numbered pickle files found in candidates: {existing_dirs}")

    latest = max(all_pickles, key=lambda p: p.stat().st_mtime)
    return load_from_pickle(latest)


def _dedupe_pickle_paths(paths: List[Path]) -> List[Path]:
    by_key: dict[str, Path] = {}
    for p in paths:
        by_key[str(p.resolve())] = p
    return list(by_key.values())


def _merge_pickles_chronologically(paths: List[Path]) -> Any:
    """
    Seřadí pickle podle mtime (starší první) a složí je přes ``_merge_results_top_level``.
    Poslední výskyt daného biasu / group_type přepíše dřívější (stejně jako postupné upserty).
    """
    if not paths:
        raise FileNotFoundError("No pickle paths to merge.")
    ordered = sorted(paths, key=lambda p: p.stat().st_mtime)
    merged: Any = None
    for p in ordered:
        data = load_from_pickle(p)
        if merged is None:
            merged = data
        elif isinstance(merged, dict) and isinstance(data, dict):
            merged = _merge_results_top_level(merged, data)
        else:
            merged = data
    return merged


def _collect_all_pickles_under_dirs(dirs: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for _, p in _list_numbered_pickles(d):
            paths.append(p)
    return _dedupe_pickle_paths(paths)


def _load_eval_pickle_merged_from_any_dir(dirs: List[Path]) -> Any:
    existing_dirs = [d for d in dirs if d.is_dir()]
    if not existing_dirs:
        raise FileNotFoundError("No candidate cache directories exist.")
    all_paths = _collect_all_pickles_under_dirs(existing_dirs)
    if not all_paths:
        raise FileNotFoundError(f"No numbered pickle files found in candidates: {existing_dirs}")
    return _merge_pickles_chronologically(all_paths)


def _latest_numbered_pickle_path(evaluation_dir: Path) -> Optional[Path]:
    numbered = _list_numbered_pickles(evaluation_dir)
    if not numbered:
        return None
    _, p = max(numbered, key=lambda x: x[0])
    return p


def _candidate_labeled_dirs(
    *,
    window_size: str,
    eval_type: Literal["train", "validation", "test"],
    evaluation_name: str,
    group_size: Optional[int],
    groups_count: Optional[int],
    group_types: Optional[Iterable[str]],
) -> List[Path]:
    candidates: List[Path] = []
    strict_metadata = (groups_count is not None) or (group_types is not None)

    # 1) exact (new layout with optional metadata)
    candidates.append(
        evaluation_results_dir(
            window_size=window_size,
            eval_type=eval_type,
            evaluation_name=evaluation_name,
            group_size=group_size,
            groups_count=groups_count,
            group_types=group_types,
            layout="labeled",
        )
    )

    # 2) Přímá cesta ``w_<W>/split_<split>/<evaluation_name>/`` (bez ``eval_n_*``).
    #    - Bez striktních metadat: jako dřív.
    #    - Jen při ``groups_count``: přidá se i tato větev po (1), protože cache často má
    #      ``eval_n_<N>`` jen u části oken; jinak tabulka končí na ``--`` pro menší W.
    if not strict_metadata or groups_count is not None:
        candidates.append(
            evaluation_results_dir(
                window_size=window_size,
                eval_type=eval_type,
                evaluation_name=evaluation_name,
                group_size=group_size,
                groups_count=None,
                group_types=None,
                layout="labeled",
            )
        )

    # 3) discovery fallback under split root (useful when caller does not pass groups_count/group_types)
    split_root = EVALUATIONS_DIR / f"w_{window_size}"
    if group_size is not None:
        split_root = split_root / f"group_{group_size}"
    split_root = split_root / f"split_{eval_type}"
    if split_root.is_dir() and not strict_metadata:
        for p in split_root.rglob(evaluation_name):
            if p.is_dir():
                candidates.append(p)
        # Všechny eval_n_<N>/<module>.py — různé GROUPS_COUNT ukládají do různých složek; bez tohoto
        # by merge tabulek (nebo load s merge_all_pickles) minul starší běhy v jiném eval_n_*.
        for sub in sorted(split_root.iterdir(), key=lambda x: x.name):
            if not sub.is_dir() or not sub.name.startswith("eval_n_"):
                continue
            ep = sub / evaluation_name
            if ep.is_dir():
                candidates.append(ep)

    # dedupe preserve order
    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        unique.append(c)
    return unique


def safe_eval_res(
    data,
    evaluation_name: str,
    window_size: str,
    type: Literal["train", "validation", "test"],
    group_size=None,
    groups_count: Optional[int] = None,
    group_types: Optional[Iterable[str]] = None,
):

    EVALUATION_DIR = evaluation_results_dir(
        window_size=window_size,
        eval_type=type,
        evaluation_name=evaluation_name,
        group_size=group_size,
        groups_count=groups_count,
        group_types=group_types,
        layout="labeled",
    )
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    existing = []
    for f in EVALUATION_DIR.glob("*.pkl"):
        try:
            existing.append(int(f.stem))  # f.stem = název bez přípony
        except ValueError:
            pass  # ignoruj soubory, co nejsou číslo

    next_num = max(existing, default=0) + 1
    file_path = EVALUATION_DIR / f"{next_num}.pkl"

    save_to_pickle(data, file_path)
    return file_path


def _stats_leaf_has_rfc_average(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    if "average" in v:
        return True
    m = v.get("metrics")
    return isinstance(m, dict) and "average" in m


def _looks_like_group_type_bias_branch(d: Any) -> bool:
    """
    Shape produced by experiments: group_type -> {bias_key -> stats_dict}.
    stats_dict has RFC ``average`` at top level or under ``metrics``.
    """
    if not isinstance(d, dict) or not d:
        return False
    for v in d.values():
        if not isinstance(v, dict):
            return False
        if _stats_leaf_has_rfc_average(v):
            return True
    return False


def _bias_inner_key_normalize(b: Any) -> Any:
    """Sloučí 0 / 0.0 / np scalar do stejného float klíče při merge bias větví."""
    if isinstance(b, bool):
        return b
    if isinstance(b, (int, float)):
        return float(b)
    if isinstance(b, str):
        try:
            return float(b)
        except ValueError:
            return b
    try:
        return float(b)
    except (TypeError, ValueError):
        return b


def _merge_results_top_level(existing: dict, incoming: dict) -> dict:
    """
    Upsert at top level (group types: similar, outlier, ...).

    For branches that look like ``{bias -> stats}``, **merge by bias key** so a run
    with only e.g. ``--population-biases 0`` does not erase previously stored biases.
    """
    merged = dict(existing) if isinstance(existing, dict) else {}
    if not isinstance(incoming, dict):
        return merged
    for k, v in incoming.items():
        prev = merged.get(k)
        if (
            isinstance(prev, dict)
            and isinstance(v, dict)
            and _looks_like_group_type_bias_branch(prev)
            and _looks_like_group_type_bias_branch(v)
        ):
            inner: dict = {}
            for bk, bv in prev.items():
                inner[_bias_inner_key_normalize(bk)] = bv
            for bk, bv in v.items():
                inner[_bias_inner_key_normalize(bk)] = bv
            merged[k] = inner
        else:
            merged[k] = v
    return merged


def upsert_eval_res(
    data,
    evaluation_name: str,
    window_size: str,
    type: Literal["train", "validation", "test"],
    group_size=None,
    groups_count: Optional[int] = None,
    group_types: Optional[Iterable[str]] = None,
):
    evaluation_dir = evaluation_results_dir(
        window_size=window_size,
        eval_type=type,
        evaluation_name=evaluation_name,
        group_size=group_size,
        groups_count=groups_count,
        group_types=group_types,
        layout="labeled",
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_numbered_pickle_path(evaluation_dir)
    if latest is None:
        out = evaluation_dir / "1.pkl"
        save_to_pickle(data, out)
        return out

    existing = load_from_pickle(latest)
    merged = _merge_results_top_level(existing, data)
    save_to_pickle(merged, latest)
    return latest


def strip_ground_truths_from_nested_pickle(path: str):
    if not os.path.isfile(path):
        print(f"⛔ Soubor neexistuje: {path}")
        return

    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(f"❌ Nelze načíst {path}: {e}")
            return

    modified = False

    for group_type, biases in data.items():
        if not isinstance(biases, dict):
            continue
        for bias_key, metrics in biases.items():
            if isinstance(metrics, dict) and "ground_truths" in metrics:
                del metrics["ground_truths"]
                modified = True

    if modified:
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"✅ Vyčištěno a přepsáno: {path}")
        except Exception as e:
            print(f"❌ Nelze přepsat {path}: {e}")
    else:
        print(f"✅ V souboru {path} nebyly žádné ground_truths – nic se nemění")

def load_eval_res(
    evaluation_name: str,
    window_size: str,
    type: Literal["train", "validation", "test"],
    num: Optional[int] = None,
    group_size=None,
    groups_count: Optional[int] = None,
    group_types: Optional[Iterable[str]] = None,
    *,
    merge_all_pickles: bool = False,
):
    """
    Načte uložený výsledek evaluace.

    Zkusí nejdřív novou pojmenovanou hierarchii (``w_*``, ``split_*``, …), pak legacy cestu,
    aby staré pickle po změně layoutu pořád šly načíst.

    Args:
        evaluation_name: podsložka = název souboru modulu experimentu (``*.py``).
        window_size: řetězec ``str(w_size)`` (consensus window).
        type: ``train`` / ``validation`` / ``test``.
        num: konkrétní ``N.pkl``; ``None`` = nejvyšší číslo v dané složce.
        group_size: u large-group evalu odpovídá segmentu ``group_<n>`` / legacy ``large/<n>``.
        groups_count: volitelný segment ``eval_n_<N>`` (nový layout).
        group_types: volitelný segment ``gtypes_<...>`` (nový layout).
        merge_all_pickles: pokud ``True`` a ``num is None``, načte **všechna** číslovaná ``*.pkl``
            ve všech kandidátních složkách (včetně ``eval_n_*`` vs. kořene) a složí je stejně jako
            upsert (sloučení biasů napříč soubory). Jinak se bere jen jeden nejnovější soubor podle
            mtime — ten může mít jen dílčí biasy a tabulky pak ukazují NaN.
    """
    errors: list[str] = []
    labeled_dirs = _candidate_labeled_dirs(
        window_size=window_size,
        eval_type=type,
        evaluation_name=evaluation_name,
        group_size=group_size,
        groups_count=groups_count,
        group_types=group_types,
    )
    try:
        if merge_all_pickles and num is None:
            return _load_eval_pickle_merged_from_any_dir(labeled_dirs)
        return _load_eval_pickle_from_any_dir(labeled_dirs, num)
    except FileNotFoundError as e:
        errors.append(f"labeled candidates failed: {e}")

    d_legacy = evaluation_results_dir(
        window_size=window_size,
        eval_type=type,
        evaluation_name=evaluation_name,
        group_size=group_size,
        layout="legacy",
    )
    try:
        if merge_all_pickles and num is None:
            return _load_eval_pickle_merged_from_any_dir([d_legacy])
        return _load_eval_pickle_from_dir(d_legacy, num)
    except FileNotFoundError as e:
        errors.append(f"legacy → {d_legacy}: {e}")

    raise FileNotFoundError(" | ".join(errors))


def autorun(experiment_cls: Type[ConsensusExperimentBase]) -> None:
    """
    CLI + execution for evaluations:
        - load last (or --num) cached results when mode auto/load
        - compute and save when missing or --mode compute
        - print LaTeX table via experiment.make_table
        - optional experiment.stats(results)
    """
    parser = build_autorun_argparser()
    args = parser.parse_args()
    if args.debug_profile:
        os.environ["CONS_EVAL_DEBUG_PROFILE"] = "1"
    experiment = experiment_cls.from_cli_args(args)
    profile_path = start_session(
        experiment.evaluation_name,
        metadata={
            "eval_type": experiment.eval_type,
            "w_size": experiment.w_size,
            "groups_count": experiment.groups_count,
            "group_size": experiment.group_size,
            "mode": args.mode,
            "save_mode": args.save_mode,
            "num": args.num,
        },
        tag=args.debug_profile_tag,
    )
    if profile_path is not None:
        print(f"[debug-profile] enabled -> {profile_path}")

    results = None
    w_key = str(experiment.w_size)

    if args.mode in ("auto", "load"):
        try:
            with timed("cache.load"):
                results = load_eval_res(
                    experiment.evaluation_name,
                    w_key,
                    experiment.eval_type,
                    num=args.num,
                    group_size=experiment.group_size,
                    groups_count=getattr(experiment, "groups_count", None),
                    group_types=getattr(experiment, "group_types", None),
                )
            print(f"[cache] Loaded {experiment.evaluation_name}/{experiment.eval_type} (num={args.num or 'last'}).")
            log_event("cache.load.hit")
        except FileNotFoundError as e:
            if args.mode == "load":
                raise
            print(f"[cache] Not found ({e}). Falling back to compute...")
            log_event("cache.load.miss", extra={"error": str(e)})

    if results is None or args.mode == "compute":
        print("[compute] Running evaluation...")
        with timed("compute.results"):
            results = experiment.compute_results()
        common_kwargs = dict(
            evaluation_name=experiment.evaluation_name,
            window_size=w_key,
            type=experiment.eval_type,
            group_size=experiment.group_size,
            groups_count=getattr(experiment, "groups_count", None),
            group_types=getattr(experiment, "group_types", None),
        )
        if args.save_mode == "upsert":
            with timed("cache.save.upsert"):
                path = upsert_eval_res(results, **common_kwargs)
            print("[cache] Upsert mode: merged by group type into latest pickle.")
        else:
            with timed("cache.save.append"):
                path = safe_eval_res(results, **common_kwargs)
            print("[cache] Append mode: wrote a new numbered pickle.")
        print(f"[cache] Saved to {path}")

    print("\n===== TABLE CODE =====")
    with timed("table.render"):
        print(experiment.make_table(results))

    stats_fn = getattr(experiment, "stats", None)
    if callable(stats_fn):
        print("\n===== STATS =====")
        try:
            with timed("stats.render"):
                stats_text = stats_fn(results)
            if stats_text is not None:
                print(stats_text)
        except Exception as e:
            print(f"[stats] Error while generating stats: {e}")
            log_event("stats.error", extra={"error": str(e)})
    if get_profile_file() is not None:
        end_session(extra={"status": "ok"})