"""
``ConsensusAgentBasedEvaluator`` — **per-group simulator** used by ``Runner.run``.

Builds on ``UserVoteSimulator``: for each group it repeatedly asks the mediator for candidate lists,
records votes, checks consensus termination, and collects optional NDCG statistics. Supports
parallel execution (thread pool by default; optional process pool on Linux via env).

Helper functions in this file also implement NDCG@k and ranking concatenation utilities shared by metrics.
"""

import os
import sys
import threading
from time import perf_counter

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.debug_profile import (
    reset_simulation_aggregates,
    sim_add_time,
    sim_flush_summary,
    sim_incr,
)
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_evaluation_agents.evaluation_agent import UserVoteSimulator
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter
import math
import numpy as np
from tqdm import tqdm
import numpy as np

# Populated only for Linux fork-based process pool (see run_simulation).
_MP_RUN_SIMULATION_STATE: Dict[str, Any] = {}


def _mp_simulate_one_worker(group: Tuple[int, ...]) -> Optional[Tuple[Tuple[int, ...], Dict[str, Any]]]:
    """Child entry point: must read state from _MP_RUN_SIMULATION_STATE (fork copy on Linux)."""
    ev: "ConsensusAgentBasedEvaluator" = _MP_RUN_SIMULATION_STATE["evaluator"]
    evaluation_factory = _MP_RUN_SIMULATION_STATE["factory"]
    t0 = perf_counter()
    mediator, _ = evaluation_factory(group)
    sim_add_time("sim.per_group.factory", perf_counter() - t0)
    t0 = perf_counter()
    simulation_result = ev._simulation_agent.simulate_group_decision(
        group, mediator, ev._end_on_first_match, ev._max_rounds
    )
    sim_add_time("sim.per_group.simulate_decision", perf_counter() - t0)
    sim_incr("sim.groups", 1)
    if simulation_result["round_found"]:
        return tuple(group), simulation_result
    return None


def _use_process_pool_for_groups(workers: int) -> bool:
    """
    ThreadPoolExecutor does not scale CPU-bound Python across cores (GIL).
    Optional Linux fork-pool runs one group per process in parallel (true multi-core).

    Enable: CONS_EVAL_USE_PROCESS_POOL=1 (Linux fork pool — skutečná multi-jádra).
    Disable: unset / 0 (default) — ThreadPoolExecutor jako dřív (GIL, málo jader).
    """
    if workers <= 1:
        return False
    if sys.platform != "linux":
        return False
    # Fork z ne-main vlákna (např. ThreadPoolExecutor v tune_* přes group_type) je nebezpečný → vynutit threads.
    if threading.current_thread() is not threading.main_thread():
        return False
    v = os.environ.get("CONS_EVAL_USE_PROCESS_POOL")
    if v is None or str(v).strip() == "":
        return False
    raw = str(v).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return raw in ("1", "true", "yes", "on")


def ndcg_at_k(ranking: List[int], ground_truth: Set[int], k: int) -> float:
    """
    NDCG@k for a single user (binary relevance).
    The ranking may be shorter than k (only the available prefix is used);
    IDCG is computed for min(k, |GT|).
    """
    if k <= 0:
        return 0.0

    # DCG
    dcg = 0.0
    for j, item in enumerate(ranking[:k], start=1):
        if item in ground_truth:
            dcg += 1.0 / math.log2(j + 1)

    # IDCG
    max_rel = min(len(ground_truth), k)
    if max_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, max_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0

def _concat_user_ranking(rounds: List[Dict[int, List[int]]], user_id: int) -> List[int]:
    """
    Concatenate per-round recommendations for a user into a single ranking,
    preserving order and removing duplicates.
    """
    ranking: List[int] = []
    seen: Set[int] = set()

    for rmap in rounds:
        round_recommendations = rmap.get(user_id, [])
        for item_id in round_recommendations:
            if item_id not in seen:
                ranking.append(item_id)
                seen.add(item_id)
    return ranking


def _first_consensus_ranks_per_user_one_based(sim_res: Dict[str, Any]) -> Optional[np.ndarray]:
    rf = sim_res.get("round_found")
    ai = sim_res.get("agreed_item")
    ar = sim_res.get("all_recommendations")
    if not rf or not ai or not ar:
        return None
    try:
        r0 = int(rf[0])
        aid = ai[0]
    except (TypeError, ValueError, IndexError):
        return None
    if r0 <= 0:
        return None
    rec_idx = r0 - 1
    if rec_idx >= len(ar):
        return None
    last_recs = ar[rec_idx]
    if not isinstance(last_recs, dict):
        return None
    positions: List[float] = []
    for _uid, items in last_recs.items():
        if not items:
            continue
        try:
            positions.append(float(items.index(aid) + 1))
        except ValueError:
            continue
    if not positions:
        return None
    return np.asarray(positions, dtype=float)


def _first_consensus_global_positions_per_user_one_based(sim_res: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Global 1-based position from the start of session until first consensus item.

    For each user:
      global_pos = items_seen_before_first_consensus_round + local_pos_in_consensus_round
    where local_pos_in_consensus_round is 1-based index of agreed item within that round.
    """
    rf = sim_res.get("round_found")
    ai = sim_res.get("agreed_item")
    ar = sim_res.get("all_recommendations")
    if not rf or not ai or not ar:
        return None
    try:
        r0 = int(rf[0])
        aid = ai[0]
    except (TypeError, ValueError, IndexError):
        return None
    if r0 <= 0:
        return None
    rec_idx = r0 - 1
    if rec_idx >= len(ar):
        return None

    positions: List[float] = []
    for _uid, rounds_items in ar[rec_idx].items():
        if not rounds_items:
            continue
        try:
            local_pos = rounds_items.index(aid) + 1  # 1-based
        except ValueError:
            continue
        items_before = 0
        for i in range(rec_idx):
            items_before += len(ar[i].get(_uid, []))
        positions.append(float(items_before + local_pos))

    if not positions:
        return None
    return np.asarray(positions, dtype=float)


def _resolve_group_gt_key(
    recommendation_user_keys: Tuple[int, ...],
    groups_ground_truth: Dict[Tuple[int, ...], Set[int]],
) -> Tuple[int, ...]:
    """
    Ground-truth dict keys follow ``tuple(group)`` as stored in eval lists (insertion order
    from simulation matches that list). ``find_group_ndcg`` used to use ``sorted`` tuples,
    which breaks lookups for the same three users in different orders.
    """
    if recommendation_user_keys in groups_ground_truth:
        return recommendation_user_keys
    sorted_key = tuple(sorted(recommendation_user_keys))
    if sorted_key in groups_ground_truth:
        return sorted_key
    members = frozenset(recommendation_user_keys)
    for k in groups_ground_truth:
        if frozenset(k) == members:
            return k
    raise KeyError(recommendation_user_keys)


def find_group_ndcg(
        group_recommendations: List[Dict[int, List[int]]],
        users_ground_truth: Dict[int, Set[int]],
        groups_ground_truth: Dict[Tuple[int, ...], Set[int]],
        k: int
    ):

    group_key = tuple(group_recommendations[0].keys())
    group_users = tuple(sorted(group_key))

    per_user_ndcg: Dict[int, float] = {}

    per_user_ndcg_com: Dict[int, float] = {}
    gt_key = _resolve_group_gt_key(group_key, groups_ground_truth)
    gt_com = groups_ground_truth[gt_key]

    for uid in group_users:
        ranking = _concat_user_ranking(group_recommendations, uid)
        gt = users_ground_truth.get(uid, set())
        per_user_ndcg[uid] = ndcg_at_k(ranking, gt, k)

        per_user_ndcg_com[uid] = ndcg_at_k(ranking, gt_com, k)

    vals = list(per_user_ndcg.values())
    vals_com = list(per_user_ndcg_com.values())

    return {
        "k": k,
        "group_users": group_users,
        "per_user_ndcg_mean": float(np.mean(vals)) if vals else 0.0,
        "per_user_ndcg_min":  float(np.min(vals))  if vals else 0.0,
        "per_user_ndcg_max":  float(np.max(vals))  if vals else 0.0,
        "per_user_ndcg": per_user_ndcg,

        "ndcg_com_mean": float(np.mean(vals_com)) if vals_com else 0.0,
        "ndcg_com_min":  float(np.min(vals_com))  if vals_com else 0.0,
        "ndcg_com_max":  float(np.max(vals_com))  if vals_com else 0.0,
        "per_user_ndcg_com": per_user_ndcg_com,
    }


def aggregate_ndcg_over_groups(
    all_recommendations: Dict[Tuple, List[Dict[int, List[int]]]],
    users_ground_truth: Dict[int, Set[int]],
    groups_ground_truth: Dict[Tuple[int, ...], Set[int]],
    k: int
) -> Dict[str, Any]:
    means, mins, maxs = [], [], []
    per_group: List[Dict[str, Any]] = []

    com_means, com_mins, com_maxs = [], [], []

    for gkey, group_recommendations in all_recommendations.items():
        group_ndcg = find_group_ndcg(group_recommendations, users_ground_truth, groups_ground_truth, k)
        group_ndcg["group_key"] = gkey
        per_group.append(group_ndcg)

        means.append(group_ndcg["per_user_ndcg_mean"])
        mins.append(group_ndcg["per_user_ndcg_min"])
        maxs.append(group_ndcg["per_user_ndcg_max"])

        com_means.append(group_ndcg["ndcg_com_mean"])
        com_mins.append(group_ndcg["ndcg_com_min"])
        com_maxs.append(group_ndcg["ndcg_com_max"])

    def safe_mean(x):
        return float(np.mean(x)) if x else 0.0

#    def safe_min(x):
#        return float(np.min(x)) if x else 0.0
#
#    def safe_max(x):
#        return float(np.max(x)) if x else 0.0

    return {
        "k": k,
        # --------- INDIVIDUAL NDCG ---------------
        "per_user_ndcg_mean_overall": safe_mean(means),
        "per_user_ndcg_min_overall": safe_mean(mins),
        "per_user_ndcg_max_overall": safe_mean(maxs),
        # --------- COM NDCG ---------------
        "ndcg_com_mean_overall": safe_mean(com_means),
        "ndcg_com_max_overall": safe_mean(com_maxs),
        "ndcg_com_min_overall": safe_mean(com_mins),
        "per_group": per_group
    }

def diag_common_hits(all_recs, users_gt, k=10):
    sizes, hits, recall = [], [], []
    for gkey, rounds in all_recs.items():
        users = list(rounds[0].keys())
        gt_com = set(users_gt.get(users[0], set()))
        for u in users[1:]:
            gt_com &= users_gt.get(u, set())
        sizes.append(len(gt_com))

        h_any = 0
        found = 0
        for u in users:
            r = []
            for rm in rounds: r.extend(rm.get(u, []))
            r = r[:k]
            # kolik z common GT je v top-K
            found += sum(1 for x in r if x in gt_com)
            if any(x in gt_com for x in r):
                h_any += 1

        hits.append(h_any/len(users) if users else 0.0)
        recall.append(found/len(gt_com) if gt_com else 0.0)

    if not sizes:
        nan = float("nan")
        return {
            "avg_common_gt_size": nan,
            "p_groups_with_nonempty_common": nan,
            "avg_user_hit_rate_into_common@k": nan,
            "avg_recall_common@k": nan,
        }

    return {
        "avg_common_gt_size": float(np.mean(sizes)),
        "p_groups_with_nonempty_common": float(np.mean([s > 0 for s in sizes])),
        "avg_user_hit_rate_into_common@k": float(np.mean(hits)),
        "avg_recall_common@k": float(np.mean(recall)),
    }

class ConsensusAgentBasedEvaluator:
    """Runs many groups through ``UserVoteSimulator`` + mediator factory; aggregates RFC/NDCG stats."""

    def __init__(self, simulation_agent: UserVoteSimulator,
                evaluation_set_type: str,
                eval_set_groups_data: str,
                groups_ground_truth: Dict[Tuple[int, ...], Set[int]],
                users_ground_truth: Dict[int, Set[int]],
                group_type: str,
                max_rounds: int = 10,
                end_on_first_match: bool = False # TRUE: evaluation ends after first match
                ):
        self._simulation_agent = simulation_agent
        self._evaluation_set_type = evaluation_set_type
        self._eval_set_groups_data = eval_set_groups_data
        self._group_type = group_type
        self._max_rounds = max_rounds
        self._end_on_first_match = end_on_first_match
        self.groups_ground_truth: Dict[Tuple[int, ...], Set[int]] = groups_ground_truth
        self.users_ground_truth: Dict[int, Set[int]] = users_ground_truth

    def run_simulation(
                    self,
                    evaluation_factory,
                    max_number_of_groups: int,
                    ndcg_k: Optional[List[int]] = None,
                    workers: int = 1,
                    ):
        reset_simulation_aggregates()
        matched_groups: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        groups = self._eval_set_groups_data[:max_number_of_groups]
        total_number_of_groups = len(groups)

        workers = max(1, int(workers or 1))

        if workers == 1:
            for group in tqdm(groups, desc="Processing groups"):
                self._simulate(group, evaluation_factory, matched_groups)
        else:
            def simulate_one(group):
                t0 = perf_counter()
                mediator, _ = evaluation_factory(group)
                sim_add_time("sim.per_group.factory", perf_counter() - t0)
                t0 = perf_counter()
                simulation_result = self._simulation_agent.simulate_group_decision(
                    group, mediator, self._end_on_first_match, self._max_rounds
                )
                sim_add_time("sim.per_group.simulate_decision", perf_counter() - t0)
                sim_incr("sim.groups", 1)
                if simulation_result["round_found"]:
                    return tuple(group), simulation_result
                return None

            def _merge_thread_results():
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(simulate_one, group) for group in groups]
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Processing groups ({workers} workers)",
                    ):
                        result = fut.result()
                        if result is None:
                            continue
                        gkey, sim_res = result
                        matched_groups[gkey] = sim_res

            if _use_process_pool_for_groups(workers):
                _MP_RUN_SIMULATION_STATE.clear()
                _MP_RUN_SIMULATION_STATE["evaluator"] = self
                _MP_RUN_SIMULATION_STATE["factory"] = evaluation_factory
                try:
                    ctx = mp.get_context("fork")
                    with ctx.Pool(processes=workers) as pool:
                        for result in tqdm(
                            pool.imap_unordered(_mp_simulate_one_worker, groups, chunksize=1),
                            total=len(groups),
                            desc=f"Processing groups ({workers} proc)",
                        ):
                            if result is None:
                                continue
                            gkey, sim_res = result
                            matched_groups[gkey] = sim_res
                except Exception as e:
                    print(
                        f"[ConsensusAgentBasedEvaluator] CONS_EVAL_USE_PROCESS_POOL fork-pool failed ({e!r}); "
                        "falling back to threads.",
                        flush=True,
                    )
                    _merge_thread_results()
                finally:
                    _MP_RUN_SIMULATION_STATE.clear()
            else:
                _merge_thread_results()

        t0 = perf_counter()
        _, metadata = evaluation_factory([])
        sim_add_time("sim.post.metadata_factory", perf_counter() - t0)

        t0 = perf_counter()
        stats = self._compute_stats(
            matched_groups,
            metadata,
            total_number_of_groups,
            ndcg_k
        )
        sim_add_time("sim.post.compute_stats", perf_counter() - t0)

        sim_flush_summary(
            {
                "workers": workers,
                "max_number_of_groups": max_number_of_groups,
                "groups_scheduled": total_number_of_groups,
                "matched_groups": len(matched_groups),
                "max_rounds": self._max_rounds,
                "end_on_first_match": self._end_on_first_match,
            }
        )
        return stats

    def _compute_stats(
                    self,
                    matched_groups,
                    mediator_metadata,
                    total_number_of_groups,
                    ndcg_k: Optional[List[int]] = None
                    ):

        print(f"found groups count: {len(matched_groups)}")
        nan = float("nan")
        if not matched_groups:
            final_stats: Dict[str, Any] = {
                "evaluation_metadata": {
                    "mediator_meta": mediator_metadata,
                    "evaluation_set": self._evaluation_set_type,
                    "group_type": self._group_type,
                    "agent_meta": self._simulation_agent.get_metadata(),
                },
                "average": nan,
                "variance": nan,
                "std_dev": nan,
                "total_rounds": total_number_of_groups,
                "counts": Counter(),
                "ratios": {},
                "matched_groups": np.array([], dtype=float),
                "first_consensus_rank_across_groups": nan,
            }
            if ndcg_k:
                ks = ndcg_k if isinstance(ndcg_k, (list, tuple)) else [ndcg_k]
                data_for_ndcg: Dict[Tuple, List[Dict[int, List[int]]]] = {}
                for k in ks:
                    ndcg_block = aggregate_ndcg_over_groups(
                        data_for_ndcg,
                        self.users_ground_truth,
                        self.groups_ground_truth,
                        k,
                    )
                    diag_block = diag_common_hits(
                        data_for_ndcg,
                        self.users_ground_truth,
                        k=k,
                    )
                    print(f"\n🔎 NDCG diagnostics for k={k}")
                    print(f"  • Avg |common GT|: {diag_block['avg_common_gt_size']}")
                    print(f"  • % groups with non-empty common: {diag_block['p_groups_with_nonempty_common']}%")
                    print(f"  • Avg user hit-rate into common@{k}: {diag_block['avg_user_hit_rate_into_common@k']}")
                    final_stats[f"ndcg@{k}"] = ndcg_block
            return final_stats

        rounds_data = np.array([v["round_found"][0] for v in matched_groups.values()])
        # first_consensus_rank_across_groups: jedno float ve final_stats; žádné velké pole do pickle.
        # first_consensus_global_position_across_groups: průměrná globální 1-based pozice
        # (kolik karet uživatel viděl do první shody; napříč všemi skupinami).
        rank_means: List[float] = []
        global_pos_means: List[float] = []
        for v in matched_groups.values():
            pos = _first_consensus_ranks_per_user_one_based(v)
            if pos is not None and pos.size > 0:
                rank_means.append(float(pos.mean()))
            gpos = _first_consensus_global_positions_per_user_one_based(v)
            if gpos is not None and gpos.size > 0:
                global_pos_means.append(float(gpos.mean()))
        rank_mean_arr = np.asarray(rank_means, dtype=float) if rank_means else np.array([], dtype=float)
        global_pos_mean_arr = (
            np.asarray(global_pos_means, dtype=float) if global_pos_means else np.array([], dtype=float)
        )

        counts = Counter(rounds_data)
        total = len(rounds_data)
        ratios = {k: v / total for k, v in counts.items()}

        final_stats = {
            "evaluation_metadata": {
                "mediator_meta": mediator_metadata,
                "evaluation_set": self._evaluation_set_type,
                "group_type": self._group_type,
                "agent_meta": self._simulation_agent.get_metadata()
            },
            "average": rounds_data.mean(),
            "variance": rounds_data.var(),
            "std_dev": rounds_data.std(),
            "total_rounds": total_number_of_groups,
            "counts": counts,
            "ratios": ratios,
            "matched_groups": rounds_data,
            "first_consensus_rank_across_groups": float(rank_mean_arr.mean())
            if rank_mean_arr.size
            else float("nan"),
            "first_consensus_global_position_across_groups": float(global_pos_mean_arr.mean())
            if global_pos_mean_arr.size
            else float("nan"),
        }

        if ndcg_k:
            ks = ndcg_k if isinstance(ndcg_k, (list, tuple)) else [ndcg_k]

            data_for_ndcg = {
                group_key: sim_res["all_recommendations"]
                for group_key, sim_res in matched_groups.items()
            }

            for k in ks:
                ndcg_block = aggregate_ndcg_over_groups(
                    data_for_ndcg,
                    self.users_ground_truth,
                    self.groups_ground_truth,
                    k
                )

                diag_block = diag_common_hits(
                    data_for_ndcg,
                    self.users_ground_truth,
                    k=k
                )

                print(f"\n🔎 NDCG diagnostics for k={k}")
                print(f"  • Avg |common GT|: {diag_block['avg_common_gt_size']:.2f}")
                print(f"  • % groups with non-empty common: {diag_block['p_groups_with_nonempty_common']*100:.1f}%")
                print(f"  • Avg user hit-rate into common@{k}: {diag_block['avg_user_hit_rate_into_common@k']:.3f}")

                # přidej pod jménem "ndcg@k"
                final_stats[f"ndcg@{k}"] = ndcg_block

        return final_stats

    # sync + async
    def _simulate(self, group, evaluation_factory, matched_groups):
        t0 = perf_counter()
        mediator, metadata = evaluation_factory(group)
        sim_add_time("sim.per_group.factory", perf_counter() - t0)

        t0 = perf_counter()
        simulation_result = self._simulation_agent.simulate_group_decision(
            group, mediator, self._end_on_first_match, self._max_rounds
        )
        sim_add_time("sim.per_group.simulate_decision", perf_counter() - t0)
        sim_incr("sim.groups", 1)

        if simulation_result["round_found"]:
            matched_groups[tuple(group)] = simulation_result