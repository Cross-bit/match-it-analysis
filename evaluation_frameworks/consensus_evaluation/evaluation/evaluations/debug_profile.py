"""
Optional **performance tracing** for consensus evaluations (off by default).

Enable with **environment** ``CONS_EVAL_DEBUG_PROFILE=1`` and/or CLI ``--debug-profile`` on experiments
(see ``config.autorun`` — it sets the env when the flag is passed). Writes append-only **JSONL** under
``cache/cons_evaluations/_debug_profiles/*.jsonl``: session boundaries, coarse stages (e.g. dataset load),
and per-group/per-round aggregates when the evaluator flushes simulation timers.

**When to use:** diagnose slow ``Runner.run`` phases, worker imbalance, or hot spots inside
``UserVoteSimulator`` — not needed for routine benchmark runs.

``timed(...)`` context managers and ``log_event`` are no-ops when profiling is disabled, so imports stay cheap.
"""

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

from utils.config import CACHE_FILES_DIR

_LOCK = threading.Lock()
_PROFILE_FILE: Optional[Path] = None
_SESSION_META: Dict[str, Any] = {}

# Aggregated timings inside group simulation (thread-safe for ThreadPoolExecutor).
_SIM_ACC_LOCK = threading.Lock()
_SIM_TIMES: Dict[str, float] = {}
_SIM_COUNTS: Dict[str, int] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_enabled() -> bool:
    return os.environ.get("CONS_EVAL_DEBUG_PROFILE", "0") == "1"


def get_profile_file() -> Optional[Path]:
    return _PROFILE_FILE


def start_session(evaluation_name: str, metadata: Optional[Dict[str, Any]] = None, tag: Optional[str] = None) -> Optional[Path]:
    global _PROFILE_FILE, _SESSION_META
    if not is_enabled():
        return None

    profiles_dir = CACHE_FILES_DIR / "cons_evaluations" / "_debug_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = evaluation_name.replace(".py", "").replace(os.sep, "_")
    suffix = f"-{tag}" if tag else ""
    profile_file = profiles_dir / f"{ts}-{safe_name}-pid{os.getpid()}{suffix}.jsonl"

    _PROFILE_FILE = profile_file
    _SESSION_META = dict(metadata or {})
    log_event("session.start", extra=_SESSION_META)
    return profile_file


def log_event(stage: str, duration_s: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    if not is_enabled() or _PROFILE_FILE is None:
        return
    payload: Dict[str, Any] = {
        "ts": _utc_now_iso(),
        "stage": stage,
    }
    if duration_s is not None:
        payload["duration_s"] = round(float(duration_s), 6)
    if extra:
        payload["extra"] = extra

    with _LOCK:
        with _PROFILE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


@contextmanager
def timed(stage: str, extra: Optional[Dict[str, Any]] = None):
    t0 = perf_counter()
    try:
        yield
    finally:
        log_event(stage, duration_s=perf_counter() - t0, extra=extra)


def end_session(extra: Optional[Dict[str, Any]] = None) -> None:
    log_event("session.end", extra=extra)


def reset_simulation_aggregates() -> None:
    if not is_enabled():
        return
    with _SIM_ACC_LOCK:
        _SIM_TIMES.clear()
        _SIM_COUNTS.clear()


def sim_add_time(key: str, seconds: float) -> None:
    if not is_enabled() or seconds <= 0.0:
        return
    with _SIM_ACC_LOCK:
        _SIM_TIMES[key] = _SIM_TIMES.get(key, 0.0) + float(seconds)


def sim_incr(key: str, n: int = 1) -> None:
    if not is_enabled() or n <= 0:
        return
    with _SIM_ACC_LOCK:
        _SIM_COUNTS[key] = _SIM_COUNTS.get(key, 0) + int(n)


def sim_flush_summary(extra: Optional[Dict[str, Any]] = None) -> None:
    """
    One JSONL row with totals + simple derived rates (mean per group / per round).
    """
    if not is_enabled() or _PROFILE_FILE is None:
        return
    with _SIM_ACC_LOCK:
        times = dict(_SIM_TIMES)
        counts = dict(_SIM_COUNTS)
    n_groups = counts.get("sim.groups", 0)
    n_rounds = counts.get("sim.rounds", 0)
    derived: Dict[str, Any] = {}
    if n_groups > 0:
        derived["mean_s_per_group"] = {
            k.replace("sim.per_group.", ""): round(times.get(k, 0.0) / n_groups, 6)
            for k in times
            if k.startswith("sim.per_group.")
        }
    if n_rounds > 0:
        derived["mean_s_per_round"] = {
            k.replace("sim.round.", ""): round(times.get(k, 0.0) / n_rounds, 6)
            for k in times
            if k.startswith("sim.round.")
        }
    n_votes = counts.get("sim.vote_calls", 0)
    if n_votes > 0 and times.get("sim.round.voting", 0.0) > 0:
        derived["mean_s_per_vote_call"] = round(times["sim.round.voting"] / n_votes, 8)
    payload = {
        "times_s": {k: round(v, 6) for k, v in sorted(times.items())},
        "counts": dict(sorted(counts.items())),
        "derived": derived,
    }
    if extra:
        payload["run"] = extra
    log_event("simulation.detail.summary", extra=payload)
