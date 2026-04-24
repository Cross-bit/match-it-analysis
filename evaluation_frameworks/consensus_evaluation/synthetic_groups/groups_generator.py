"""
`synthetic_groups.groups_generator` — synthetic user triplets for consensus evaluation.

Builds **groups of user IDs** (typically size 3) used as evaluation scenarios: similar
users in embedding space, outliers, random controls, and specialised variants
(divergent / high rating-variance on common items). The core class is
``GroupGenerator``:

- normalises user embeddings and precomputes **FAISS** nearest neighbours (cosine
  via inner product on L2-normalised vectors),
- estimates global similarity percentiles ``ts`` / ``to`` to classify "similar" vs
  "dissimilar" pairs,
- optionally intersects candidate groups with **sparse rating overlap** filters
  (minimum common rated items).

Helper functions train or load a **LightFM** model on the MovieLens CSR matrix,
cache **precision@k** probes, and derive user embedding dicts — all via
``load_or_build_pickle`` so long offline steps are reproducible.

The ``if __name__ == "__main__"`` block at the bottom is an **offline batch driver**
(ml-32m): it materialises large pickle caches for similar / outlier / random groups
and filtered variants; it is not used when the package is imported by evaluators.
"""

from collections import defaultdict
from multiprocessing import Pool, cpu_count
from overrides import override
from tqdm import tqdm
import random
import faiss
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from scipy.sparse import csr_matrix
from dataset.data_access import MovieLensDatasetLoader
from utils.config import load_or_build_pickle

# Toggles for the ``__main__`` batch job (dataset / cache artefacts).
LOAD_SPARSE_MATRIX = True
LOAD_SURPRISE_TRAINSET = True
LOAD_EMBEDDINGS_FROM_PICKLE = True


def train_or_load_lightfm_model(csr_data, *, model_name="lightfm-ml-32m.pkl", dims=50, epochs=10, loss="warp"):

    def train_model():
        model = LightFM(no_components=dims, loss=loss, random_state=42)
        model.fit(csr_data, epochs=epochs, num_threads=4)
        return model

    model = load_or_build_pickle(
        model_name,
        train_model,
        description=f"LightFM model ({loss}, {dims} dims, {epochs} epochs)"
    )

    return model

def evaluate_precision_light_fm_cached(model, csr_matrix, *, k=10, cache_name=None, description=None):
    """
    Evaluate precision@k and cache the result using load_or_build_pickle.
    """

    def compute_precision():
        prec = precision_at_k(model, csr_matrix, k=k, num_threads=4)
        return prec

    if cache_name is None:
        cache_name = f"light_fm_ml_32m_precision_k{k}.pkl"

    if description is None:
        description = f"Precision@{k}"

    return load_or_build_pickle(cache_name, compute_precision, description=description)

def create_user_embeddings_lightfm(model: LightFM, user_id_map: Dict = None) -> Dict[int, np.ndarray]:
    """
    Extract user embeddings from a trained LightFM model.

    Args:
        model: Trained LightFM instance.
        user_id_map: Optional internal index → external user id mapping.

    Returns:
        Mapping ``{user_id: embedding_vector}``.
    """
    n_users = model.user_embeddings.shape[0]
    user_embeddings = {}

    for user_idx in range(n_users):
        user_id = user_id_map[user_idx] if user_id_map else user_idx
        user_embeddings[user_id] = model.user_embeddings[user_idx]

    return user_embeddings

class GroupGenerator:
    def __init__(
        self,
        user_embeddings: Dict[int, np.ndarray],
        ts_percentile: int = 75,
        to_percentile: int = 25,
        max_neighbors: int = 200,
        load_percentiles = False
    ):
        """
        Synthetic groups generator using FAISS (fast cosine similarity).
        """
        self.user_ids = list(user_embeddings.keys())
        self.id_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.index_to_id = {idx: uid for idx, uid in enumerate(self.user_ids)}

        # Embedding matrix as float32 + L2 row normalisation (unit vectors for cosine IP).
        self.embeddings = np.array([user_embeddings[uid] for uid in self.user_ids], dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        self.embeddings = self.embeddings / norms

        # CACHE
        faiss_cache_name = f"faiss-nn-{len(self.user_ids)}users-{max_neighbors}.pkl"


        def build_faiss_neighbors():
            print("⚙️  Fitting FAISS index...")
            index = faiss.IndexFlatIP(self.embeddings.shape[1])  # IP == cosine similarity on L2-normalised rows
            index.add(self.embeddings)
            distances, indices = index.search(self.embeddings, max_neighbors)
            return distances, indices

        self.similarities, self.indices = load_or_build_pickle(
            faiss_cache_name,
            build_faiss_neighbors,
            description="FAISS cosine neighbors"
        )

        percentile_sample = 5_000_000
        percentile_estimates_cache_name = f"percentiles-estimates-faiss-model-{faiss_cache_name}-{percentile_sample}-to-{to_percentile}-ts-{ts_percentile}.pkl"

        self.ts, self.to  = load_or_build_pickle(
            percentile_estimates_cache_name,
            lambda: self._compute_global_similarity_percentile(self.embeddings, ts_percentile, to_percentile, num_samples=percentile_sample, seed=42, workers=42, chunk_size=20000),
            description="Groups percentiles estimates"
        )

        print(self.ts, self.to)

    def _sample_similarity_worker(self, args: Tuple[np.ndarray, int]) -> float:
        embeddings, seed = args
        np.random.seed(seed)
        i, j = np.random.choice(len(embeddings), size=2, replace=False)
        return float(np.dot(embeddings[i], embeddings[j]))

    def _compute_global_similarity_percentile(
        self,
        embeddings: np.ndarray,
        percentile_ts: float,
        percentile_to: float,
        num_samples: int = 1_000_000,
        workers: int = 8,
        chunk_size: int = 10000,
        seed: int = 42
    ) -> float:
        """
        Estimate a global similarity threshold (e.g. 25th percentile) from random pairs.

        Args:
            embeddings (np.ndarray): Normalized user embeddings (n_users, dim)
            percentile (float): e.g. 25 for 25th percentile
            num_samples (int): Number of random pairs to sample

        Returns:
            float: Estimated percentile similarity value
        """
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1_000_000_000, size=num_samples)

        args = [(embeddings, s) for s in seeds]

        with Pool(processes=workers or cpu_count()) as pool:
            similarities = list(
                tqdm(
                    pool.imap_unordered(self._sample_similarity_worker, args, chunksize=chunk_size),
                    total=num_samples,
                    desc="📊 Sampling global similarities (parallel)"
                )
            )

        ts = float(np.percentile(similarities, percentile_ts))
        to = float(np.percentile(similarities, percentile_to))

        return (ts, to)

    def generate_random_group(self, groups_count: int, group_size: int = 3) -> List[List[int]]:
        """Generate unique random groups of given size.

        Args:
            groups_count (int): Number of groups to generate.
            group_size (int): Size of each group.

        Returns:
            List[List[int]]: List of unique user ID groups.
        """
        groups = []
        seen_groups = set()
        users = np.array(self.user_ids)

        max_attempts = groups_count * 10
        attempts = 0

        while len(groups) < groups_count and attempts < max_attempts:
            group = tuple(sorted(np.random.choice(users, size=group_size, replace=False)))

            if group not in seen_groups:
                seen_groups.add(group)
                groups.append(list(group))

            attempts += 1

        if len(groups) < groups_count:
            print(f"⚠️ Warning: Only {len(groups)} unique groups could be generated (limit reached).")

        return groups

    def generate_similar_group(self, groups_count: int, group_size: int = 3) -> List[List[int]]:
        groups = []
        seen_groups = set()

        for i, seed_user_id in enumerate(self.user_ids):
            if len(groups) >= groups_count:
                break

            group = [seed_user_id]

            # Get neighbors sorted by similarity (already precomputed)
            neighbor_indices = self.indices[i][1:]  # skip self
            neighbor_sims = self.similarities[i][1:]

            for idx, sim in zip(neighbor_indices, neighbor_sims):
                candidate_id = self.index_to_id[idx]
                if sim < self.ts:
                    continue

                # Check if candidate is similar to all current group members
                if all(
                    self._similarity_between_users(candidate_id, other_id) >= self.ts
                    for other_id in group
                ):
                    group.append(candidate_id)

                if len(group) == group_size:
                    break

            if len(group) == group_size:
                key = tuple(sorted(group))
                if key not in seen_groups:
                    seen_groups.add(key)
                    groups.append(group)

        return groups


    def _similarity_between_users(self, uid1: int, uid2: int) -> float:
        """Return cosine similarity between two user IDs using precomputed embeddings."""
        v1 = self.embeddings[self.id_to_index[uid1]]
        v2 = self.embeddings[self.id_to_index[uid2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    def generate_outlier_group(self, groups_count: int) -> List[List[int]]:
        """Generate groups with two similar users and one outlier using FAISS-based similarities.

        Args:
            groups_count (int): Number of groups to generate.

        Returns:
            List[List[int]]: Groups where two users are similar and one is dissimilar (outlier).
        """
        groups = []
        seen_groups = set()

        for i, seed_user_id in enumerate(self.user_ids):
            if len(groups) >= groups_count:
                break

            # Step 1: find one similar user (skip self)
            neighbor_indices = self.indices[i][1:]
            neighbor_sims = self.similarities[i][1:]

            similar_users = [
                self.index_to_id[idx]
                for idx, sim in zip(neighbor_indices, neighbor_sims)
                if sim >= self.ts
            ]

            if not similar_users:
                continue

            user_b = np.random.choice(similar_users)

            # Step 2: find one dissimilar user (both to A and B)
            seed_index_b = self.id_to_index[user_b]

            # Sample dissimilar candidates
            for _ in range(1000):  # limit attempts
                candidate_idx = np.random.randint(len(self.user_ids))
                candidate_id = self.index_to_id[candidate_idx]

                if candidate_id in {seed_user_id, user_b}:
                    continue

                sim_to_a = self._similarity_between_users(seed_user_id, candidate_id)
                sim_to_b = self._similarity_between_users(user_b, candidate_id)

                if sim_to_a <= self.to and sim_to_b <= self.to:
                    group = [seed_user_id, user_b, candidate_id]
                    group_key = tuple(sorted(group))
                    if group_key not in seen_groups:
                        seen_groups.add(group_key)
                        groups.append(group)
                    break  # go to next seed

        return groups

    def filter_groups_with_common_items(
        self,
        groups: List[List[int]],  # user IDs (external, e.g., MovieLens)
        ratings_csr: csr_matrix,
        user_id_map: Dict[int, int],  # real_user_id -> csr_index
        min_common_items: int = 5
    ) -> List[List[int]]:

        filtered = []

        for group in groups:
            try:
                user_indices = [user_id_map[uid] for uid in group]
            except KeyError:
                continue  # skip group if any user is not in the map

            if any(u >= ratings_csr.shape[0] for u in user_indices):
                continue  # just in case – safety check

            rated_sets = [set(ratings_csr[u].indices) for u in user_indices]
            common_items = set.intersection(*rated_sets)

            if len(common_items) >= min_common_items:
                filtered.append(group)  # return original external IDs

        return filtered

class GroupGeneratorRestrictedInteractions(GroupGenerator):

    def __init__(self, ratings_csr: csr_matrix, user_id_map: Dict[int, int], user_embeddings, ts_percentile = 75, to_percentile = 25, max_neighbors = 200):
        """
            csr_matrix (csr_matrix): User-item interaction matrix (filtered CSR).
            user_id_map (Dict[int, int]): Mapping of internal user IDs to original MovieLens IDs.
        """
        super().__init__(user_embeddings, ts_percentile, to_percentile, max_neighbors)
        self._user_id_map = user_id_map
        self._ratings_csr = ratings_csr

    @override
    def generate_random_group(
        self,
        groups_count: int,
        group_size: int = 3,
        min_common_items: int = 5,
    ) -> List[List[int]]:
        """
        Generate unique random groups of given size that share at least `min_common_items`.

        Args:
            groups_count (int): Number of groups to generate.
            group_size (int): Size of each group.
            min_common_items (int): Minimum number of shared items per group.


        Returns:
            List[List[int]]: List of valid user ID groups.
        """

        external_to_internal_id = {external_id : csr_index for csr_index, external_id in self._user_id_map.items()}

        groups: List[List[int]] = []
        seen_groups: Set[Tuple[int, ...]] = set()
        users = np.array(self.user_ids)
        rng = np.random.default_rng(42)

        # Build row views once; avoids expensive set(...) allocations in every attempt.
        row_items = {}
        for uid in users:
            try:
                idx = external_to_internal_id[int(uid)]
            except KeyError:
                continue
            row = self._ratings_csr[idx]
            row_items[int(uid)] = row.indices  # CSR indices are sorted and unique

        max_attempts = groups_count * 50
        attempts = 0

        progress = tqdm(total=groups_count, desc="🎲 Generating random groups")

        while len(groups) < groups_count and attempts < max_attempts:
            group = tuple(sorted(rng.choice(users, size=group_size, replace=False)))
            if group in seen_groups:
                attempts += 1
                continue

            try:
                arrays = [row_items[int(uid)] for uid in group]
            except KeyError:
                attempts += 1
                continue

            # Fast progressive intersection with early stop.
            common = arrays[0]
            for arr in arrays[1:]:
                common = np.intersect1d(common, arr, assume_unique=True)
                if common.size < min_common_items:
                    break

            if common.size >= min_common_items:
                seen_groups.add(group)
                groups.append(list(group))
                progress.update(1)

            attempts += 1

        progress.close()
        if len(groups) < groups_count:
            print(f"⚠️ Warning: Only {len(groups)} unique groups could be generated (limit reached).")

        return groups

    def generate_similar_group(
        self,
        groups_count: int,
        group_size: int = 3,
        min_common_items: int = 5
    ) -> List[List[int]]:
        external_to_index = {external_id : csr_index for csr_index, external_id in self._user_id_map.items()}
        groups = []
        seen_groups = set()

        progress = tqdm(total=groups_count, desc="✅ Found similar groups")

        for i, seed_user_id in enumerate(self.user_ids):
            if len(groups) >= groups_count:
                break

            group = [seed_user_id]

            neighbor_indices = self.indices[i][1:]
            neighbor_sims = self.similarities[i][1:]

            for idx, sim in zip(neighbor_indices, neighbor_sims):
                candidate_id = self.index_to_id[idx]
                if sim < self.ts:
                    continue

                if all(
                    self._similarity_between_users(candidate_id, other_id) >= self.ts
                    for other_id in group
                ):
                    group.append(candidate_id)

                if len(group) == group_size:
                    break

            if len(group) == group_size:
                try:
                    user_indices = [external_to_index[uid] for uid in group]
                    user_item_sets = [set(self._ratings_csr[u].indices) for u in user_indices]
                    common_items = set.intersection(*user_item_sets)

                    if len(common_items) >= min_common_items:
                        key = tuple(sorted(group))
                        if key not in seen_groups:
                            seen_groups.add(key)
                            groups.append(group)
                            progress.update(1)

                except KeyError:
                    continue

        progress.close()
        return groups

    def generate_outlier_group_from_similar(
        self,
        similar_groups: List[List[int]],
        min_common_items: int = 5,
        sample_candidates: int = 6000,
        groups_count: int = 10000
    ) -> List[List[int]]:
        """
        Generate outlier groups of size 3 by extending existing similar groups of 3 with a dissimilar user.
        Outlier must be dissimilar to both group members and share enough common items.
        """
        reverse_map = {v: k for k, v in self._user_id_map.items()}
        all_user_ids = set(self.user_ids)
        groups = []
        seen_groups = set()

        # Track only successful group creations
        pbar = tqdm(total=groups_count, desc="👽 Found outlier groups")

        for group in similar_groups:
            if len(groups) >= groups_count:
                break

            # Generate all pairs from group
            pairs = [(group[i], group[j]) for i in range(3) for j in range(i + 1, 3)]

            for user_a, user_b in pairs:
                if len(groups) >= groups_count:
                    break

                try:
                    a_idx = reverse_map[user_a]
                    b_idx = reverse_map[user_b]
                except KeyError:
                    continue

                candidate_pool = list(all_user_ids - {user_a, user_b})
                sampled_candidates = random.sample(candidate_pool, min(sample_candidates, len(candidate_pool)))

                for candidate_id in sampled_candidates:
                    try:
                        c_idx = reverse_map[candidate_id]
                    except KeyError:
                        continue

                    sim_to_a = np.dot(self.embeddings[self.id_to_index[user_a]], self.embeddings[self.id_to_index[candidate_id]])
                    sim_to_b = np.dot(self.embeddings[self.id_to_index[user_b]], self.embeddings[self.id_to_index[candidate_id]])

                    if sim_to_a > self.to or sim_to_b > self.to:
                        continue

                    items_a = set(self._ratings_csr[a_idx].indices)
                    items_b = set(self._ratings_csr[b_idx].indices)
                    items_c = set(self._ratings_csr[c_idx].indices)

                    common_ab = items_a & items_b & items_c
                    if len(common_ab) < min_common_items:
                        continue

                    triplet = tuple(sorted([user_a, user_b, candidate_id]))
                    if triplet in seen_groups:
                        continue

                    seen_groups.add(triplet)
                    groups.append(list(triplet))
                    pbar.update(1)
                    break  # valid group found → break candidate loop

        pbar.close()
        return groups

    def generate_divergent_group(
        self,
        groups_count: int,
        group_size: int = 3,
        min_common_items: int = 5,
        sample_candidates: int = 12000,
        max_attempts: int = 50_000_000,
    ) -> List[List[int]]:
        """
        Triplets where every pair has cosine similarity <= self.to (same cutoff as outlier
        dissimilarity). Candidates are drawn by **random sampling** over all users (like
        ``generate_outlier_group``), not from FAISS top-neighbors — those neighbors are
        biased toward *high* similarity, so users with sim <= ``to`` rarely appear there.
        """
        if group_size != 3:
            raise ValueError("divergent groups are only defined for group_size=3")

        reverse_map = {v: k for k, v in self._user_id_map.items()}
        groups: List[List[int]] = []
        seen: Set[Tuple[int, int, int]] = set()

        all_ids = self.user_ids
        n_users = len(all_ids)
        rng = np.random.default_rng(42)

        pbar = tqdm(total=groups_count, desc="🔻 Divergent groups")
        attempts = 0

        while len(groups) < groups_count and attempts < max_attempts:
            attempts += 1
            seed_a = all_ids[int(rng.integers(n_users))]

            user_b = None
            for _ in range(sample_candidates):
                b = all_ids[int(rng.integers(n_users))]
                if b == seed_a:
                    continue
                if self._similarity_between_users(seed_a, b) > self.to:
                    continue
                user_b = b
                break
            if user_b is None:
                continue

            user_c = None
            for _ in range(sample_candidates):
                c = all_ids[int(rng.integers(n_users))]
                if c in (seed_a, user_b):
                    continue
                if self._similarity_between_users(seed_a, c) > self.to:
                    continue
                if self._similarity_between_users(user_b, c) > self.to:
                    continue
                user_c = c
                break
            if user_c is None:
                continue

            triplet = tuple(sorted([seed_a, user_b, user_c]))
            if triplet in seen:
                continue

            try:
                a_idx = reverse_map[seed_a]
                b_idx = reverse_map[user_b]
                c_idx = reverse_map[user_c]
            except KeyError:
                continue

            items_a = set(self._ratings_csr[a_idx].indices)
            items_b = set(self._ratings_csr[b_idx].indices)
            items_c = set(self._ratings_csr[c_idx].indices)
            if len(items_a & items_b & items_c) < min_common_items:
                continue

            seen.add(triplet)
            groups.append(list(triplet))
            pbar.update(1)

        pbar.close()
        if len(groups) < groups_count:
            print(
                f"⚠️ Warning: only {len(groups)} divergent groups "
                f"(target {groups_count}) after {attempts} attempts."
            )
        return groups

    def _group_common_items_avg_rating_variance(
        self,
        group: List[int],
        reverse_map: Dict[int, int],
        min_common_items: int,
    ) -> Tuple[int, float]:
        """Return (#common_items, avg variance over common-item ratings) for a group."""
        try:
            internal = [reverse_map[u] for u in group]
        except KeyError:
            return 0, 0.0

        item_sets: List[Set[int]] = []
        user_rating_maps: List[Dict[int, float]] = []
        for idx in internal:
            row = self._ratings_csr[idx]
            items = row.indices
            vals = row.data
            item_sets.append(set(items))
            user_rating_maps.append({int(i): float(v) for i, v in zip(items, vals)})

        common = set.intersection(*item_sets)
        if len(common) < min_common_items:
            return len(common), 0.0

        vars_: List[float] = []
        for item in common:
            item_vals = [m[item] for m in user_rating_maps]
            vars_.append(float(np.var(item_vals)))
        if not vars_:
            return len(common), 0.0
        return len(common), float(np.mean(vars_))

    def generate_variance_group(
        self,
        groups_count: int,
        group_size: int = 3,
        min_common_items: int = 5,
        min_avg_item_variance: float = 1.2,
        candidate_multiplier: int = 10,
        max_candidates: int = 400_000,
    ) -> List[List[int]]:
        """
        Similar in latent space (all pairwise sims >= ts), but high rating disagreement on
        common items: average per-item variance across group members >= min_avg_item_variance.
        """
        if group_size != 3:
            raise ValueError("variance groups are only defined for group_size=3")

        reverse_map = {v: k for k, v in self._user_id_map.items()}
        candidate_target = min(max_candidates, max(groups_count * candidate_multiplier, groups_count))

        similar_candidates = self.generate_similar_group(
            groups_count=candidate_target,
            group_size=group_size,
            min_common_items=min_common_items,
        )

        scored: List[Tuple[float, Tuple[int, int, int]]] = []
        seen: Set[Tuple[int, int, int]] = set()
        pbar = tqdm(total=len(similar_candidates), desc="📈 Scoring variance groups")
        for g in similar_candidates:
            key = tuple(sorted(g))
            if key in seen:
                pbar.update(1)
                continue
            seen.add(key)

            _, avg_var = self._group_common_items_avg_rating_variance(
                group=list(key),
                reverse_map=reverse_map,
                min_common_items=min_common_items,
            )
            if avg_var >= min_avg_item_variance:
                scored.append((avg_var, key))
            pbar.update(1)
        pbar.close()

        scored.sort(key=lambda x: x[0], reverse=True)
        groups = [list(g) for _, g in scored[:groups_count]]

        if len(groups) < groups_count:
            print(
                f"⚠️ Warning: only {len(groups)} variance groups "
                f"(target {groups_count}) for min_avg_item_variance={min_avg_item_variance}."
            )
        return groups


if __name__ == "__main__":

    print("Loading data")

    data_loader = MovieLensDatasetLoader("ml-32m")
    movies_data_df, ratings_csr, user_id_map, movie_id_map = data_loader.load_sparse_ratings(LOAD_SPARSE_MATRIX)

    model = train_or_load_lightfm_model(ratings_csr)

    for i in range(10, 60, 10):
        train_precision_measures = evaluate_precision_light_fm_cached(model, ratings_csr, k=i)
        print(f"📊 Mean precision@{i}:", train_precision_measures.mean())

    users_embeddings = load_or_build_pickle("users-embeddings-50-light-fm-ml-32m.pkl",
                        lambda: create_user_embeddings_lightfm(model, user_id_map),
                        description="user embeddings LightFm"
                        )

    ##
    ## Find test groups raw
    ##

    groupsGen = GroupGenerator(users_embeddings, 75, 25)

    similar_groups_count = 100000
    outlier_groups_count = 100000
    random_groups_count = 100000

    similar_groups = load_or_build_pickle(
        f"groups-similar-ml-32m-{similar_groups_count}.pkl",
        lambda: groupsGen.generate_similar_group(groups_count=similar_groups_count, group_size=3),
        description="similar groups"
    )

    outlier_groups = load_or_build_pickle(
        f"groups-outliers-ml-32m-{outlier_groups_count}.pkl",
        lambda: groupsGen.generate_outlier_group(groups_count=outlier_groups_count),
        description="outlier groups"
    )

    random_groups = load_or_build_pickle(
        f"groups-random-ml-32m-{random_groups_count}.pkl",
        lambda: groupsGen.generate_random_group(groups_count=random_groups_count, group_size=3),
        description="random groups"
    )

    print(f"Number of groups:")
    print(f"similar {len(similar_groups)}")
    print(f"outlier {len(outlier_groups)}")
    print(f"random {len(random_groups)}")


    ##
    ## Find test groups only with top 5 closest items
    ##

    min_common_items = 5

    similar_groups_filtered = load_or_build_pickle(
        f"groups-similar-ml-32m-{similar_groups_count}-min-common-items-{min_common_items}.pkl",
        lambda: groupsGen.filter_groups_with_common_items(
        similar_groups,
        ratings_csr=ratings_csr,
        user_id_map={uid: idx for idx, uid in enumerate(user_id_map.values())},
        min_common_items=min_common_items),
        description=f"filtered similar groups -- min intersection {min_common_items}"
    )


    random_groups_filtered = load_or_build_pickle(
        f"groups-random-ml-32m-{random_groups_count}-min-common-items-{min_common_items}.pkl",
        lambda: groupsGen.filter_groups_with_common_items(
        random_groups,
        ratings_csr=ratings_csr,
        user_id_map={uid: idx for idx, uid in enumerate(user_id_map.values())},
        min_common_items=min_common_items),
        description=f"filtered random groups -- min intersection {min_common_items}"
    )


    outlier_groups_filtered = load_or_build_pickle(
        f"groups-outlier-ml-32m-{outlier_groups_count}-min-common-items-{min_common_items}.pkl",
        lambda: groupsGen.filter_groups_with_common_items(
        outlier_groups,
        ratings_csr=ratings_csr,
        user_id_map={uid: idx for idx, uid in enumerate(user_id_map.values())},
        min_common_items=min_common_items),
        description=f"filtered outlier groups -- min intersection {min_common_items}"
    )

    print(f"Number of groups after filtr top {min_common_items}:")
    print(f"similar {len(similar_groups_filtered)}")
    print(f"outlier {len(outlier_groups_filtered)}")
    print(f"random {len(random_groups_filtered)}")
