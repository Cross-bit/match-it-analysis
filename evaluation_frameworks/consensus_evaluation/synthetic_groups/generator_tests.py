"""
`generator_tests` — sanity checks for synthetic group lists and related matrices.

Utilities here validate that generated **user groups** (lists of external user
ids) satisfy structural constraints used downstream: e.g. minimum intersection of
rated items inside a CSR slice, global uniqueness of group tuples, and optional
checks against a LightFM model / embeddings for plausibility.

These are **offline QA helpers** (not pytest modules): call them from notebooks or
batch scripts after ``groups_generator`` materialises new pickle pools.
"""

from itertools import combinations
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import csr_matrix
from tqdm import tqdm


def validate_groups_min_interactions_and_uniques(
    groups: List[List[int]],
    csr_matrix: csr_matrix,
    user_id_map: Dict[int, int],
    min_interactions: int = 5
) -> Tuple[int, int, int]:
    """
    Validates the group list to ensure:
    1. Each group has at least `min_interactions` common items.
    2. All groups are unique.

    Args:
        groups (List[List[int]]): List of groups (MovieLens user IDs).
        csr_matrix (csr_matrix): Filtered binary CSR matrix.
        user_id_map (Dict[int, int]): Mapping from row index to MovieLens user ID.
        min_interactions (int): Minimum number of common items required.

    Returns:
        Tuple: (valid_groups_count, duplicate_groups_count, total_groups_count)
    """
    internal_id_map = {v: k for k, v in user_id_map.items()}
    seen: Set[Tuple[int, ...]] = set()
    valid_count = 0
    duplicate_count = 0

    for group in tqdm(groups, desc="🔎 Validating groups"):
        try:
            internal_ids = [internal_id_map[uid] for uid in group]
        except KeyError:
            continue  # Skip group if any user is missing

        user_items = [set(csr_matrix[uid].indices) for uid in internal_ids]
        common_items = set.intersection(*user_items)

        group_key = tuple(sorted(group))
        if group_key in seen:
            duplicate_count += 1
            continue

        seen.add(group_key)

        if len(common_items) >= min_interactions:
            valid_count += 1

    return valid_count, duplicate_count, len(groups)


def validate_similar_groups_users_sim_percentile(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    similarity_threshold: float,
) -> Tuple[int, int, int]:
    """
    Validates that all pairs of users in each group have similarity >= threshold.

    Args:
        groups (List[List[int]]): Groups of MovieLens user IDs.
        user_embeddings (Dict[int, np.ndarray]): Map from MovieLens user ID → embedding (normalized).
        similarity_threshold (float): Threshold similarity (e.g. 75th percentile).

    Returns:
        Tuple[int, int, int]: (valid_groups_count, invalid_groups_count, total_groups_count)
    """
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        return np.dot(u, v)

    valid_count = 0
    invalid_count = 0

    for group in tqdm(groups, desc="🧪 Validating similarity threshold"):
        try:
            embeddings = [user_embeddings[uid] for uid in group]
        except KeyError:
            invalid_count += 1
            continue

        is_valid = True
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim < similarity_threshold:
                    is_valid = False
                    break
            if not is_valid:
                break

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    return valid_count, invalid_count, len(groups)


def validate_outlier_groups_similarity(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    ts: float,
    to: float
) -> Tuple[int, int]:
    """
    Validates outlier groups according to similarity thresholds.
    Automatically detects the most similar pair and treats the third as a candidate outlier.

    Args:
        groups: List of [user1, user2, user3]
        user_embeddings: Raw (unnormalized) user embedding vectors {user_id: vector}
        ts: similarity threshold for similar users (e.g., 0.26)
        to: similarity threshold for outlier (e.g., -0.12)

    Returns:
        Tuple: (valid_groups_count, total_groups_checked)
    """
    # Normalize embeddings
    normalized = {
        uid: vec / (np.linalg.norm(vec) + 1e-10)
        for uid, vec in user_embeddings.items()
    }

    valid = 0

    for group in tqdm(groups, desc="🧪 Validating outlier group similarities"):
        try:
            a, b, c = group
            emb_a = normalized[a]
            emb_b = normalized[b]
            emb_c = normalized[c]
        except KeyError:
            continue

        # Compute all pairwise similarities
        sim_ab = np.dot(emb_a, emb_b)
        sim_ac = np.dot(emb_a, emb_c)
        sim_bc = np.dot(emb_b, emb_c)

        # Find most similar pair
        sims = [(a, b, sim_ab), (a, c, sim_ac), (b, c, sim_bc)]
        sims.sort(key=lambda x: x[2], reverse=True)

        user1, user2, sim_similar = sims[0]
        outlier = ({a, b, c} - {user1, user2}).pop()

        emb_out = normalized[outlier]
        sim1 = np.dot(emb_out, normalized[user1])
        sim2 = np.dot(emb_out, normalized[user2])

        if sim_similar >= ts and sim1 <= to and sim2 <= to:
            valid += 1

    return valid, len(groups)


def validate_similar_groups_similarity(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    ts: float
) -> Tuple[int, int]:
    """
    Validates similar groups: all user pairs in each group must have similarity >= ts.

    Args:
        groups: List of [user1, user2, user3, ...] groups
        user_embeddings: Raw (unnormalized) user embedding vectors {user_id: vector}
        ts: similarity threshold for all user pairs in the group

    Returns:
        Tuple: (valid_groups_count, total_groups_checked)
    """
    # Normalize embeddings
    normalized = {
        uid: vec / (np.linalg.norm(vec) + 1e-10)
        for uid, vec in user_embeddings.items()
    }

    valid = 0

    for group in tqdm(groups, desc="🧪 Validating similar group similarities"):
        try:
            # Get normalized embeddings
            embs = [normalized[uid] for uid in group]
        except KeyError:
            continue

        all_above_ts = all(
            np.dot(embs[i], embs[j]) >= ts
            for i, j in combinations(range(len(embs)), 2)
        )

        if all_above_ts:
            valid += 1

    return valid, len(groups)


def validate_divergent_groups_similarity(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    to: float,
) -> Tuple[int, int]:
    """
    Each group must have all pairwise cosine similarities <= to (same cutoff as outlier
    dissimilarity), on normalized embeddings.
    """
    normalized = {
        uid: vec / (np.linalg.norm(vec) + 1e-10)
        for uid, vec in user_embeddings.items()
    }

    valid = 0

    for group in tqdm(groups, desc="🧪 Validating divergent-group similarities"):
        if len(group) != 3:
            continue
        try:
            embs = [normalized[uid] for uid in group]
        except KeyError:
            continue

        sim_ab = float(np.dot(embs[0], embs[1]))
        sim_ac = float(np.dot(embs[0], embs[2]))
        sim_bc = float(np.dot(embs[1], embs[2]))

        if sim_ab <= to and sim_ac <= to and sim_bc <= to:
            valid += 1

    return valid, len(groups)


# --- CLI-style wrappers (print progress / summaries) ---


def validate_groups_min_interactions_run_wrapper(
    groups: List[List[int]],
    csr_matrix: csr_matrix,
    user_id_map: Dict[int, int],
    min_interactions = 5
):
    """Wrapper with nice print to evaluate user groups min inter

    Args:
        groups (List[List[int]]): _description_
        csr_matrix (csr_matrix): _description_
        user_id_map (Dict[int, int]): _description_
    """
    print(f"🧪 Checking min interactions (>={min_interactions} common items)")
    valid, duplicates, total = validate_groups_min_interactions_and_uniques(
        groups=groups,
        csr_matrix=csr_matrix,
        user_id_map=user_id_map,
        min_interactions=min_interactions
    )

    print(f"✅ Valid groups (>={min_interactions} common items): {valid}")
    print(f"⚠️ Duplicate groups: {duplicates}")
    print(f"📦 Total checked: {total}")


def validate_similar_groups_users_sim_percentile_run_wrapper(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    similarity_threshold: float
):
    """Wrapper with nice print to evaluate user groups min inter

    Args:
        groups (List[List[int]]): _description_
        csr_matrix (csr_matrix): _description_
        user_id_map (Dict[int, int]): _description_
    """
    print(f"🧪 Checking similar group users are all from {similarity_threshold} using cosine similarity.")

    valid, invalid, total = validate_similar_groups_users_sim_percentile(
        groups=groups,
        user_embeddings=user_embeddings,
        similarity_threshold=similarity_threshold  # např. 75. percentil
    )

    print(f"✅ Valid: {valid} ❌ Invalid: {invalid} 📦 Total: {total}")


def validate_outlier_groups_similarity_wrapper(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    ts: float,
    to: float
) -> Tuple[int, int]:

    valid, total = validate_outlier_groups_similarity(
        groups=groups,
        user_embeddings=user_embeddings,  # nenormalizované!
        ts=ts,
        to=to
    )
    print(f"✅ Valid: {valid} / {total}")

def validate_similar_groups_similarity_wrapper(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    ts: float
) -> Tuple[int, int]:

    valid, total = validate_similar_groups_similarity(
        groups=groups,
        user_embeddings=user_embeddings,  # nenormalizované!
        ts=ts
    )
    print(f"✅ Valid: {valid} / {total}")


def validate_divergent_groups_similarity_wrapper(
    groups: List[List[int]],
    user_embeddings: Dict[int, np.ndarray],
    to: float,
) -> None:
    valid, total = validate_divergent_groups_similarity(
        groups=groups,
        user_embeddings=user_embeddings,
        to=to,
    )
    print(f"✅ Divergent valid (all pairs <= to): {valid} / {total}")