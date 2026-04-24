from collections import defaultdict
import math
import random
from typing import DefaultDict, Dict, List, Optional, Set, Tuple
from lightfm import LightFM
import multiprocessing as mp
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from tqdm import tqdm
from dataset.data_access import MovieLensDatasetLoader
from utils.config import load_or_build_pickle
from scipy.sparse import csr_matrix


# ====================================
# DESCRIPTION
# ====================================
# Utility functions for computing and filtering ground truth items
# shared among group members, and for modifying CSR matrices accordingly.
#

def zero_out_from_dict(
    ratings: csr_matrix,
    removal_dict: Dict[Tuple[int, int, int], Set[int]],
    user_id_map: Dict[int, int],
) -> csr_matrix:
    """
    """
    ratings = ratings.tocsr(copy=True)

    reversed_user_id_map = { external_id : csr_index for csr_index, external_id in user_id_map.items() }

    for external_user_id_triplet, item_ids in removal_dict.items():
        for external_user_id in external_user_id_triplet:

            internal_user_id = reversed_user_id_map[external_user_id]

            row_start = ratings.indptr[internal_user_id]
            row_end = ratings.indptr[internal_user_id + 1]

            user_items = ratings.indices[row_start:row_end]
            user_data = ratings.data[row_start:row_end]

            for idx_in_row, item_id in enumerate(user_items):
                if item_id in item_ids:
                    user_data[idx_in_row] = 0.0

    ratings.eliminate_zeros()
    return ratings


def get_group_common_items(groups: List[List[int]], ratings_csr: csr_matrix, user_id_map: Dict[int, int]) -> Dict[Tuple[int, int, int], Set[int]]:
    """ Return common items for each user triplet.

    Args:
        groups (List[List[int]]): Groups triplets (as external user ids).
        ratings_csr (csr_matrix): Rating matrix of all groups members.
        user_id_map (Dict[int, int]): Mapping of internal user IDs to external IDs.

    Returns:
        Dict[Tuple[int, int, int], Set[int]]: Mapping of groups to set of common items.
    """

    reverse_map = { external_id : csr_index for csr_index, external_id in user_id_map.items() }

    result = {}
    for group in groups:
        indices = [reverse_map[uid] for uid in group]
        common = set(ratings_csr[indices[0]].indices)
        for idx in indices[1:]:
            common &= set(ratings_csr[idx].indices)
        result[tuple(group)] = common
    return result

def get_all_common_items(group_to_common_items: Dict[Tuple[int, int, int], Set[int]]) -> Set[int]:
    """_summary_

    Args:
        group_to_common_items (Dict[Tuple[int, int, int], Set[int]]): _description_

    Returns:
        Set[int]: _description_
    """

    all_items = set()
    for item_set in group_to_common_items.values():
        all_items |= item_set
    return all_items

def remap_common_items_to_external(
    common_items_dict: Dict[Tuple[int, int, int], Set[int]],
    item_id_map: Dict[int, int]
) -> Dict[Tuple[int, int, int], Set[int]]:
    """
    Remaps common item indices from internal CSR indices back to external item IDs.

    Args:
        common_items_dict: Mapping of group to set of item indices (CSR internal).
        item_id_map: Mapping of internal item indices to external item IDs.

    Returns:
        Dict with the same group keys but item sets remapped to external IDs.
    """

    remapped = {
        group: {item_id_map[item_idx] for item_idx in item_set}
        for group, item_set in common_items_dict.items()
    }
    return remapped

def _compute_target_test_size(n_interactions: int, ratio: float = 0.5, rounding: str = "ceil") -> int:
    """50/50 split; u lichých počtů interakcí použij 'ceil' (>=50 %) nebo 'floor' (<=50 %)."""
    val = n_interactions * ratio
    return math.ceil(val) if rounding == "ceil" else math.floor(val)

def _row_items_set(csr: csr_matrix, internal_user_id: int) -> Set[int]:
    start, end = csr.indptr[internal_user_id], csr.indptr[internal_user_id + 1]
    return set(csr.indices[start:end])

def zero_out_user_items(
    ratings: csr_matrix,
    removal_user_to_internal_items: Dict[int, Set[int]],  # EXTERNAL user id -> {INTERNAL item idx}
    user_id_map: Dict[int, int],
) -> csr_matrix:
    """
    Vynuluje (odstraní) zadané položky z řádků uživatelů. Předpokládá INTERNAL item idx v hodnotách.
    """
    ratings = ratings.tocsr(copy=True)
    reversed_user_id_map = { external_id : csr_index for csr_index, external_id in user_id_map.items() }

    for external_user_id, internal_item_ids in removal_user_to_internal_items.items():
        if not internal_item_ids:
            continue
        internal_user_id = reversed_user_id_map[external_user_id]
        row_start = ratings.indptr[internal_user_id]
        row_end = ratings.indptr[internal_user_id + 1]
        user_items = ratings.indices[row_start:row_end]
        user_data = ratings.data[row_start:row_end]

        # membership O(1)
        to_remove = internal_item_ids
        for idx_in_row, item_id in enumerate(user_items):
            if item_id in to_remove:
                user_data[idx_in_row] = 0.0

    ratings.eliminate_zeros()
    return ratings

def _compute_target_test_size(n_interactions: int, ratio: float = 0.5, rounding: str = "ceil") -> int:
    """50/50 split; u lichých počtů interakcí použij 'ceil' (>=50 %) nebo 'floor' (<=50 %)."""
    val = n_interactions * ratio
    return math.ceil(val) if rounding == "ceil" else math.floor(val)

def filter_groups_by_user_train_capacity(
    groups: List[List[int]],
    ratings_csr: csr_matrix,
    user_id_map: Dict[int, int],
    test_ratio: float = 0.5,
    rounding: str = "ceil",
    seed: int = 42,
) -> Tuple[
    Dict[Tuple[int, ...], Set[int]],                  # group_to_required_common_internal (capped) -- filtered groups
    Set[int],                                         # dropped_users (EXTERNAL)
    List[Tuple[int, ...]],                            # dropped_groups
]:
    """
    Single-pass filter: drop any group that contains a user whose union of 'required common'
    across all their groups would exceed their test budget (e.g., 50% of their interactions).

    Steps:
        1) Compute common items for each group.
        2) Cap each group's common set to the smallest member's test budget (min_cap).
        3) Build per-user union of these required common sets.
        4) If any user violates their budget, drop ALL groups containing them (single pass).
        5) Return kept groups and their capped required-common sets.
    """
    rng = random.Random(seed)
    ext2int_user = { ext: i for i, ext in user_id_map.items() }

    # Early exit if no groups
    if not groups:
        return {}, set(), []

    # Precompute per-user test target size (e.g., 50% of their interactions)
    users_external_ids = {u for g in groups for u in g}
    user_to_target: Dict[int, int] = {}
    for u_external_id in users_external_ids:
        user_idx = ext2int_user[u_external_id]
        n_interactions = ratings_csr.indptr[user_idx + 1] - ratings_csr.indptr[user_idx]  # count non-zeros
        user_to_target[u_external_id] = _compute_target_test_size(n_interactions, test_ratio, rounding)

    remaining_groups = [tuple(g) for g in groups]

    # 1) Common items for all remaining groups
    groups_common = get_group_common_items([list(g) for g in remaining_groups], ratings_csr, user_id_map)

    # 2) Cap common by the smallest member's test budget
    groups_common_required_internal: Dict[Tuple[int, ...], Set[int]] = {}
    for g in remaining_groups:
        min_cap = min(user_to_target[u] for u in g)
        commons = groups_common[g]
        if len(commons) <= min_cap:
            Sg = set(commons)  # take all
        else:
            Sg = set(rng.sample(list(commons), min_cap))  # sample exactly min_cap
        groups_common_required_internal[g] = Sg

    # 3) Per-user union of required common across all their groups
    per_user_required: Dict[int, Set[int]] = defaultdict(set)
    for g, Sg in groups_common_required_internal.items():
        for u in g:
            per_user_required[u].update(Sg)

    # 4) Users whose required-common union exceeds their test budget
    violators = {u for u, req in per_user_required.items() if len(req) > user_to_target[u]}

    if not violators:
        # No one exceeds the budget: keep everything
        return groups_common_required_internal, set(), []

    # Drop all groups containing any violator (single pass)
    to_drop = [g for g in remaining_groups if any(u in violators for u in g)]
    kept_groups = [g for g in remaining_groups if g not in to_drop]

    # Keep required-common only for the kept groups
    groups_common_required_internal_kept: Dict[Tuple[int, ...], Set[int]] = {
        g: groups_common_required_internal[g] for g in kept_groups
    }

    dropped_groups: List[Tuple[int, ...]] = to_drop
    dropped_users: Set[int] = set(violators)

    return groups_common_required_internal_kept, dropped_users, dropped_groups

def build_user_test_split(
    groups: List[List[int]],
    ratings_csr: csr_matrix,
    user_id_map: Dict[int, int],      # INTERNAL user idx -> EXTERNAL user id
    item_id_map: Dict[int, int],      # INTERNAL item idx -> EXTERNAL item id
    precomputed_required_common_internal: Dict[Tuple[int, ...], Set[int]],
    test_ratio: float = 0.5,
    rounding: str = "ceil",
    seed: int = 42
) -> Tuple[
    Dict[int, Set[int]],  # user_to_test_items_internal : EXTERNAL user id -> {INTERNAL item idx}
    Dict[int, Set[int]],  # user_to_test_items_external : EXTERNAL user id -> {EXTERNAL item id}
    Dict[Tuple[int, ...], Set[int]],  # group_to_required_common_internal
    Dict[Tuple[int, ...], Set[int]],  # group_to_required_common_external
]:
    """
    Z předfiltrovaných skupin sestaví per-user test tak, aby:
    - obsahoval povinné common (předané v 'precomputed_required_common_internal' nebo nově spočítané),
    - byl doplněn do 50 % profilu.
    Pokud po filtru stále nastane situace, že required > 50 % u některého uživatele, vyhodíme ValueError.
    """
    rng = random.Random(seed)
    ext2int_user = { ext : i for i, ext in user_id_map.items() }
    int2ext_item = item_id_map

    users_ext = {u for g in groups for u in g}

    # Precompute per-user test target size (e.g., 50% of their interactions)
    users_external_ids = {u for g in groups for u in g}
    user_to_target: Dict[int, int] = {}
    for u_external_id in users_external_ids:
        user_idx = ext2int_user[u_external_id]
        n_interactions = ratings_csr.indptr[user_idx + 1] - ratings_csr.indptr[user_idx]  # count non-zeros
        user_to_target[u_external_id] = _compute_target_test_size(n_interactions, test_ratio, rounding)

    g2required = precomputed_required_common_internal

    # agregace povinných common per user
    required_per_user: Dict[int, Set[int]] = defaultdict(set)
    for g, Sg in g2required.items():
        for u in g:
            required_per_user[u].update(Sg)

    # Verification --> this should hold
    violators = {u for u, S in required_per_user.items() if len(S) > user_to_target[u]}
    if violators:
        raise ValueError(f"Po filtru stále existují uživatelé s required_common > 50%: {violators}")

    # fill the user ground truth
    user_to_test_internal: Dict[int, Set[int]] = {}
    for u_external_id in users_ext:

        u_internal_id = ext2int_user[u_external_id]

        Iu = _row_items_set(ratings_csr, u_internal_id) # all users items
        required_common = set(required_per_user[u_external_id]) # all the group common items

        target_items_count = user_to_target[u_external_id] # number of items user should have

        remaining = list(Iu - required_common) # remove the group GT from the user items
        k = max(0, target_items_count - len(required_common))

        if k > len(remaining): # for users with not enough items
            k = len(remaining)

        sampled = set(rng.sample(remaining, k)) if k > 0 else set() # sample all the remaining items
        user_to_test_internal[u_external_id] = required_common | sampled

    user_to_test_external: Dict[int, Set[int]] = {
        u_ext: { int2ext_item[i] for i in items_internal }
        for u_ext, items_internal in user_to_test_internal.items()
    }

    g2required_external: Dict[Tuple[int, ...], Set[int]] = {
        g: { int2ext_item[i] for i in Sg }
        for g, Sg in g2required.items()
    }

    return user_to_test_internal, user_to_test_external, g2required, g2required_external

def prepare_group_eval_data2_test_split(groups: List[List[int]], ratings_csr: csr_matrix, user_id_map: Dict[int, int], item_id_map: Dict[int, int], test_ratio=0.5) -> Tuple[csr_matrix, List[List[int]], Dict[int, Set[int]], Dict[Tuple[int, ...], Set[int]]]:
    """
        Prepares evaluation data for group recommendation using a train/test split.

        For each user, a portion of their interactions is held out as test data
        (according to `test_ratio`), and the remaining interactions are kept in the
        filtered CSR matrix. The function also builds group-level ground-truth items
        based on external IDs.

        Args:
            groups (List[List[int]]): List of groups of users (external user IDs).
            ratings_csr (csr_matrix): Original user-item interaction matrix.
            user_id_map (Dict[int, int]): Mapping from external user IDs to internal indices.
            item_id_map (Dict[int, int]): Mapping from internal item IDs to external item IDs.
            test_ratio (float, optional): Fraction of each user's items to hold out for testing.
                                        Defaults to 0.5.

        Returns:
            Tuple:
                - csr_matrix: Filtered CSR matrix with test interactions removed
                            (i.e., training-only interactions).
                - Dict[int, Set[int]]: Mapping of each external user ID to their held-out
                                    test items (external item IDs).
                - Dict[Tuple[int, ...], Set[int]]: Mapping of each group (tuple of external user IDs)
                                                to their common ground-truth items (external item IDs).
    """

    # 1) Keep groups and get their capped common (INTERNAL ids)
    group_to_required_common_internal, dropped_users, dropped_groups = filter_groups_by_user_train_capacity(groups, ratings_csr, user_id_map, test_ratio=0.5)
    print(f"✅ Filtered only valid users with enough items; dropped_groups: {len(dropped_groups)}, dropped_users: {len(dropped_users)}, groups_left: {len(group_to_required_common_internal)}")

    # 2) Build per-user TEST and zero-out TEST from CSR
    kept_groups = [list(g) for g in group_to_required_common_internal.keys()]

    user_to_test_internal, user_to_test_external, _, group_to_required_common_external = build_user_test_split(
        groups=kept_groups,
        ratings_csr=ratings_csr,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        precomputed_required_common_internal=group_to_required_common_internal,
        test_ratio=test_ratio,
    )

    print(f"✅ Created user test split")

    filtered_csr = zero_out_user_items(
        ratings=ratings_csr,
        removal_user_to_internal_items=user_to_test_internal,
        user_id_map=user_id_map,
    )

    return (
        filtered_csr, # csr without test data
        kept_groups,
        user_to_test_external, # external user id TO external item test ids (e.g. the 50 % of all the user data)
        group_to_required_common_external, # group (composed of external ids) TO users common group items
    )


def prepare_group_eval_data(groups: List[List[int]], ratings_csr: csr_matrix, user_id_map: Dict[int, int], item_id_map: Dict[int, int]) -> Tuple[csr_matrix, Dict[Tuple[int, int, int], Set[int]]]:
    """
    Prepares a test CSR matrix for group evaluation by removing ground truth items.
    It removes all the ground truth items for all the groups.

                Args:
        groups (List[List[int]]): List of user triplets used for evaluation (should contain external users IDs e.g. from MovieLens).
        ratings_csr (csr_matrix): Original user-item interaction matrix.
        user_id_map (Dict[int, int]): Mapping from internal index user IDs to external ids.

    Returns:
        Tuple:
            - csr_matrix: Filtered CSR matrix with ground-truth (common) items removed.
            - Dict[Tuple[int, int, int], Set[int]]: Mapping of each group to its common ground-truth items.
    """
    common_items_dict: Dict[Tuple[int, int, int], Set[int]] = get_group_common_items(groups, ratings_csr, user_id_map)
    print("✅ Found groups common (ground-truth) interactions")
    filtered_csr = zero_out_from_dict(ratings_csr, common_items_dict, user_id_map)
    remapped_common = remap_common_items_to_external(common_items_dict, item_id_map)
    print("✅ Remapped common items to external item IDs")

    return (filtered_csr, remapped_common)
