"""
Průběh evaluace v dávce (např. ``run_consensus_eval_fast.sh``).

Bash (volitelné, pro řádek „globálně x/y“):
  CONS_EVAL_BATCH_MODULE_IDX / CONS_EVAL_BATCH_MODULE_TOTAL
  CONS_EVAL_BATCH_GRAND_OFFSET — součet kroků dokončených před tímto modulem
  CONS_EVAL_BATCH_GRAND_TOTAL  — součet přes celou dávku (musí sedět s pořadím modulů)

Před každým ``Runner.run``:
  ``set_progress_slot_before_runner(slot_1based, n_slots, n_biases)``

``n_slots`` je počet volání ``run`` v modulu (typicky len(group_types), tj. 5).

Hlavní řádek ukazuje **krok uvnitř modulu** (slot×bias); globální zlomek jen pokud
bash nastaví GRAND_* (jinak dřívější vzorec ``mt*ns*nb`` lhal, když ``ns`` mezi moduly není stejné).
"""

from __future__ import annotations

import os


def progress_disabled() -> bool:
    return os.environ.get("CONS_EVAL_PROGRESS_DISABLE", "").strip() in ("1", "true", "yes")


def set_progress_slot_before_runner(slot_index_1based: int, n_slots: int, n_biases: int) -> None:
    """Nastav env před ``Runner.run`` (worker i hlavní vlákno)."""
    os.environ["CONS_EVAL_PROGRESS_SLOT_IDX"] = str(int(slot_index_1based))
    os.environ["CONS_EVAL_PROGRESS_N_SLOTS"] = str(int(n_slots))
    os.environ["CONS_EVAL_PROGRESS_N_BIAS"] = str(int(n_biases))


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def print_bias_completed_global_progress(group_type: str, bias_idx_1based: int, n_bias: int) -> None:
    """Vola se po dokončení simulace+NDCG pro jeden population bias."""
    if progress_disabled():
        return

    mi = _read_int("CONS_EVAL_BATCH_MODULE_IDX", 1)
    mt = _read_int("CONS_EVAL_BATCH_MODULE_TOTAL", 1)
    slot = _read_int("CONS_EVAL_PROGRESS_SLOT_IDX", 1)
    ns = _read_int("CONS_EVAL_PROGRESS_N_SLOTS", 1)
    nb = _read_int("CONS_EVAL_PROGRESS_N_BIAS", n_bias)
    if nb < 1:
        nb = max(1, n_bias)

    if mi < 1:
        mi = 1
    if mt < 1:
        mt = 1
    if slot < 1:
        slot = 1
    if ns < 1:
        ns = 1

    within_cur = (slot - 1) * nb + bias_idx_1based
    within_tot = ns * nb
    within_left = max(0, within_tot - within_cur)

    grand = os.environ.get("CONS_EVAL_BATCH_GRAND_TOTAL", "").strip()
    off = _read_int("CONS_EVAL_BATCH_GRAND_OFFSET", 0)
    global_hint = ""
    if grand.isdigit() and int(grand) > 0:
        gcur = off + within_cur
        global_hint = f" · globálně {gcur}/{int(grand)}"

    print(
        f"\n[cons_eval průběh] modul {mi}/{mt} · v tomto modulu {within_cur}/{within_tot} "
        f"(zbývá v modulu ~{within_left} · slot {slot}/{ns} · bias {bias_idx_1based}/{nb} "
        f"· group_type={group_type!r}){global_hint}\n",
        flush=True,
    )


def print_runner_batch_preamble(group_type: str, n_bias: int) -> None:
    """Krátká hlavička na začátku ``Runner.run`` (kolik jednotek v tomto modulu)."""
    if progress_disabled():
        return

    mi = _read_int("CONS_EVAL_BATCH_MODULE_IDX", 1)
    mt = _read_int("CONS_EVAL_BATCH_MODULE_TOTAL", 1)
    ns = _read_int("CONS_EVAL_PROGRESS_N_SLOTS", 1)
    nb = _read_int("CONS_EVAL_PROGRESS_N_BIAS", max(1, n_bias))
    if nb < 1:
        nb = max(1, n_bias)
    if ns < 1:
        ns = 1

    mod_units = ns * nb
    grand = os.environ.get("CONS_EVAL_BATCH_GRAND_TOTAL", "").strip()
    gextra = ""
    if grand.isdigit():
        gextra = f" · odhad dávky celkem {int(grand)} kroků (viz CONS_EVAL_BATCH_GRAND_TOTAL)"
    print(
        f"[cons_eval plán] tento modul: {mod_units} kroků (slots={ns} × biasy={nb}) "
        f"· modul {mi}/{mt} v dávce · aktuální group_type={group_type!r}{gextra}",
        flush=True,
    )
