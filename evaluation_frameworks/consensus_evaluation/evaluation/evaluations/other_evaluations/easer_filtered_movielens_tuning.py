import random
import math
import numpy as np
from typing import Dict, List, Set, Tuple, Any
from tqdm import tqdm

from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.model_train_load import train_or_load_easer_model
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluation_context_factory import build_context_holdout

# =========================================
# CONFIG
# =========================================
EVAL_SET = "test"          # "train" | "validation" | "test"
GROUP_TYPE = "similar"     # "similar" | "outlier" | "random" | "divergent"
UNIQUE_GROUPS = True

LIMIT_GROUPS = 1000        # kolik skupin vyhodnotit (None = všechny)
SAMPLE_RANDOM = False      # True = náhodný výběr, False = prvních N
SEED = 42                  # jen když SAMPLE_RANDOM = True

L2S = [7500, 10000, 15000]
KS  = [5, 10, 20]
EXCLUDE_TRAIN_SEEN = True  # doporučeno: nedoporučovat tréninkem viděné položky
CAST_B_TO_FLOAT32 = True   # půlí paměť pro B

# =========================================
# NDCG (bez debug výpisů)
# =========================================
def ndcg_at_k(ranking: List[int], ground_truth: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = 0.0
    for j, item in enumerate(ranking[:k], start=1):
        if item in ground_truth:
            dcg += 1.0 / math.log2(j + 1)
    max_rel = min(len(ground_truth), k)
    if max_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, max_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0

def _concat_user_ranking(rounds: List[Dict[int, List[int]]], user_id: int) -> List[int]:
    out: List[int] = []
    for rmap in rounds:
        recs = rmap.get(user_id, [])
        if recs:
            out.extend(recs)
    return out

def find_group_ndcg(
    group_recommendations: List[Dict[int, List[int]]],
    users_ground_truth: Dict[int, Set[int]],
    groups_ground_truth: Dict[Tuple[int, ...], Set[int]],
    k: int
) -> Dict[str, Any]:
    group_users = tuple(sorted(group_recommendations[0].keys()))
    per_user_ndcg: Dict[int, float] = {}
    per_user_ndcg_com: Dict[int, float] = {}
    gt_com = groups_ground_truth.get(group_users, set())

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
    com_means, com_mins, com_maxs = [], [], []
    per_group: List[Dict[str, Any]] = []

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

    return {
        "k": k,
        "per_user_ndcg_mean_overall": float(np.mean(means)) if means else 0.0,
        "per_user_ndcg_min_overall":  float(np.mean(mins))  if mins  else 0.0,
        "per_user_ndcg_max_overall":  float(np.mean(maxs))  if maxs  else 0.0,
        "ndcg_com_mean_overall": float(np.mean(com_means)) if com_means else 0.0,
        "ndcg_com_max_overall":  float(np.mean(com_maxs))  if com_maxs  else 0.0,
        "ndcg_com_min_overall":  float(np.mean(com_mins))  if com_mins  else 0.0,
        "per_group": per_group
    }

# =========================================
# Paměťově úsporné doporučení bez cache
# =========================================
def recommend_topk_nocache(easer, uid: int, k: int, *, exclude_train_seen: bool) -> List[int]:
    """
    Spočítá skóre pro uživatele (1*N CSR @ B), volitelně zamaskuje tréninkové položky
    a vrátí Top-K externích item_id. NEVYLUČUJEME testové GT!
    """
    # mapování user -> interní index
    uidx = easer._user_id_to_internal_row_index[uid]
    row = easer._ratings_matrix[uidx, :]            # csr (1 × n_items)

    # skóre: (1×N) @ (N×N) -> (1×N)
    scores = row @ easer.B
    scores = np.asarray(scores).ravel()

    # volitelně vyřadit tréninkové položky
    if exclude_train_seen:
        seen_idx = row.indices
        if seen_idx.size:
            scores[seen_idx] = -np.inf

    # robustně omez k
    k = min(k, scores.size)
    if k == scores.size:
        top_idx = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -k)[-k:]
        top_idx = part[np.argsort(scores[part])[::-1]]

    # převod na externí item_id
    to_item_id = easer._internal_col_index_to_item_id
    return [to_item_id[int(i)] for i in top_idx]

# =========================================
# Hlavní běh
# =========================================
def main():
    # 1) Eval kontext (CSR už nemá testové interakce — jsou vyhozené do GT)
    context = build_context_holdout(EVAL_SET, GROUP_TYPE, UNIQUE_GROUPS)

    train_set = context["filtered_evaluation_set_csr"]
    user_id_map = context["user_id_map"]
    item_id_map = context["item_id_map"]
    users_ground_truth = context["users_ground_truth"]          # user -> set(external item_id)
    groups_ground_truth = context["groups_ground_truth"]        # tuple(sorted users) -> set(external item_id)
    eval_groups = list(context["eval_set_groups_data"])         # list[tuple(user_ids)]

    # 2) Omez skupiny
    orig_n = len(eval_groups)
    if LIMIT_GROUPS is not None and LIMIT_GROUPS < orig_n:
        if SAMPLE_RANDOM:
            random.seed(SEED)
            eval_groups = random.sample(eval_groups, LIMIT_GROUPS)
        else:
            eval_groups = eval_groups[:LIMIT_GROUPS]
    print(f"Evaluating on {len(eval_groups)} groups (out of original {orig_n}).")

    Ks = sorted(set(KS))
    Kmax = max(Ks)

    # 3) Vyhodnocení pro různé L2
    for l2 in L2S:
        print(f"\n=== Evaluating EASER with l2={l2} ===")

        easer = train_or_load_easer_model(
            train_set,
            user_id_map,
            item_id_map,
            model_name=f"EASER_l2_{l2}",
            regularization=l2
        )

        # volitelně zmenši paměť matice B
        if CAST_B_TO_FLOAT32:
            try:
                easer.B = easer.B.astype(np.float32, copy=False)
            except Exception:
                pass

        # 4) Generování doporučení (bez cache; NEvylučuj test GT!)
        all_recommendations: Dict[Tuple[int, ...], List[Dict[int, List[int]]]] = {}

        for group in tqdm(eval_groups, desc=f"Recommending (l2={l2})", unit="group"):
            rec_map: Dict[int, List[int]] = {}
            for uid in group:
                rec_map[uid] = recommend_topk_nocache(
                    easer, uid, Kmax, exclude_train_seen=EXCLUDE_TRAIN_SEEN
                )
            all_recommendations[tuple(group)] = [rec_map]  # 1 "kolo" doporučení na skupinu

        # 5) Výpočet metrik
        for k in Ks:
            res = aggregate_ndcg_over_groups(
                all_recommendations,
                users_ground_truth,
                groups_ground_truth,
                k=k
            )
            print(
                f"NDCG@{k}: "
                f"user-mean={res['per_user_ndcg_mean_overall']:.4f}, "
                f"common-mean={res['ndcg_com_mean_overall']:.4f}"
            )

if __name__ == "__main__":
    main()
