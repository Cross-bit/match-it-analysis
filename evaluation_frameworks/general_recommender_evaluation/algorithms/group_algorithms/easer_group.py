import numpy as np
from typing import Callable, Dict, List, Optional, Set, Tuple
from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoFull
from evaluation_frameworks.general_recommender_evaluation.iterators.top_k_iterator import TopKIterator
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.interface import RecAlgoGroupAggregated

# ===================================
#  DESCRIPTION
# ===================================
# Group recommendation algorithm.
# Extension of single-user recommender to group recommendation
# using classical strategies.
#

class GR_AggregatedRecommendations(RecAlgoGroupAggregated):

    def __init__(
                self,
                single_user_recommender: RecAlgoFull,
                custom_aggregation: Optional[Dict[str, Callable[[np.ndarray], float]]] = None):
        """
            Args:
                single_user_recommender (RecAlgoFull): A recommender supporting user vectors, item score vectors, and ID mapping.
                custom_aggregation (Optional[Dict[str, Callable]]): Optional dictionary of custom aggregation functions
                    keyed by method name.
        """

        self._single_user_recommender = single_user_recommender
        self._custom_aggregation = custom_aggregation or {}

    def predict_group(self, user_ids: List[int], item_id: int, method: str = 'mean') -> float:
        """
        Predicts a group score for a single item by aggregating individual user scores.

        Args:
            user_ids (List[int]): List of user IDs in the group.
            item_id (int): External item ID (to be mapped to internal index).
            method (str): Aggregation method. Supported: 'mean', 'min', 'max', 'median', or custom.

        Returns:
            float: Aggregated score for the group and item.
        """

        item_index = self._single_user_recommender.item_id_to_index(item_id)

        item_scores = []
        for uid in user_ids:
            user_vec = self._single_user_recommender.get_user_vector(uid)
            scores = self._single_user_recommender.get_item_scores(user_vec)
            item_scores.append(scores[item_index])

        # item_scores = list of floats
        return self._aggregate(np.array(item_scores)[np.newaxis, :], method)[0]

    def top_k_iterator(self, user_ids: List[int], method: str = 'mean', exclude: Optional[Set[int]] = None) -> TopKIterator:
        """
        Returns an iterator over top-K item recommendations for a group of users.

        Args:
            user_ids (List[int]): List of user IDs in the group.
            method (str): Aggregation method across users.
            exclude (Optional[Set[int]]): Optional set of item IDs to exclude from recommendations.

        Returns:
            TopKIterator: Iterator over (item_id, score) pairs in descending order of score.
        """
        if user_ids == []:
            return

        score_matrix = []
        for uid in user_ids:
            user_vec = self._single_user_recommender.get_user_vector(uid)
            scores = self._single_user_recommender.get_item_scores(user_vec)
            score_matrix.append(scores)

        score_matrix = np.vstack(score_matrix)
        group_scores = self._aggregate(score_matrix, method)

        item_score_pairs = [
            (
                self._single_user_recommender.index_to_item_id(idx),
                group_scores[idx]
            )
            for idx in range(len(group_scores))
        ]

        return TopKIterator(item_score_pairs, exclude=exclude)

    def _aggregate(self, score_matrix: np.ndarray, method: str) -> np.ndarray:
        """
        Aggregates a user-item score matrix across users using the specified method.

        Args:
            score_matrix (np.ndarray): Matrix of shape (n_users, n_items)
            method (str): Aggregation method: mean, min, max, median, or custom.

        Returns:
            np.ndarray: Aggregated score vector (shape: [n_items])
        """

        if method == 'mean':
            return np.mean(score_matrix, axis=0)
        elif method == 'min':
            return np.min(score_matrix, axis=0)
        elif method == 'max':
            return np.max(score_matrix, axis=0)
        elif method == 'median':
            return np.median(score_matrix, axis=0)

        elif method in ('multiplicative', 'geomean'):
        # Geometric mean — penalizes low scores.
        # Assumes nonnegative scores; epsilon avoids log(0).
            eps = 1e-12
            X = np.clip(score_matrix, eps, None)
            return np.exp(np.mean(np.log(X), axis=0))
        elif method == 'plurality':
            top = np.argmax(score_matrix, axis=1)  # (n_users,)
            n_items = score_matrix.shape[1]
            return np.bincount(top, minlength=n_items)
        elif method in self._custom_aggregation:
            return self._custom_aggregation[method](score_matrix)
        else:
            raise ValueError(f"Unsupported aggregation method '{method}'")


class GR_AggregatedProfiles(RecAlgoGroupAggregated):

    def __init__(
                self,
                single_user_recommender: RecAlgoFull,
                custom_aggregation: Optional[Dict[str, Callable[[np.ndarray], float]]] = None):
        """
            Group recommender that allows dynamic updates.

            Args:
                single_user_recommender (RecAlgoFull): A recommender supporting user vectors, item score vectors, and ID mapping.
                custom_aggregation (Optional[Dict[str, Callable]]): Optional dictionary of custom aggregation functions
                    keyed by method name.
        """

        self._single_user_recommender = single_user_recommender
        self._custom_aggregation = custom_aggregation or {}

    def predict_group(self, user_ids: List[int], item_id: int, method: str = 'mean') -> float:
        """
        Predicts a group score for a single item by aggregating individual user scores.

        Args:
            user_ids (List[int]): List of user IDs in the group.
            item_id (int): External item ID (to be mapped to internal index).
            method (str): Aggregation method. Supported: 'mean', 'min', 'max', 'median', or custom.

        Returns:
            float: Aggregated score for the group and item.
        """

        group_profile = self._aggregate_profiles(user_ids, method)
        scores = self._single_user_recommender.get_item_scores(group_profile)
        item_index = self._single_user_recommender.item_id_to_index(item_id)
        return scores[item_index]

    def top_k_iterator(self, user_ids: List[int], method: str = 'mean', exclude: Optional[Set[int]] = None) -> TopKIterator:
        """
        Returns an iterator over top-K item recommendations for a group of users.

        Args:
            user_ids (List[int]): List of user IDs in the group.
            method (str): Aggregation method across users.
            exclude (Optional[Set[int]]): Optional set of item IDs to exclude from recommendations.

        Returns:
            TopKIterator: Iterator over (item_id, score) pairs in descending order of score.
        """
        group_profile = self._aggregate_profiles(user_ids, method)
        scores = self._single_user_recommender.get_item_scores(group_profile)

        item_score_pairs = [
            (self._single_user_recommender.index_to_item_id(idx), score)
            for idx, score in enumerate(scores)
        ]

        return TopKIterator(item_score_pairs, exclude=exclude)

    def _aggregate_profiles(self, user_ids: List[int], method) -> np.ndarray:
        profiles = []
        for uid in user_ids:
            profiles.append(self._single_user_recommender.get_user_vector(uid))
        profiles = np.vstack(profiles)

        if method == "mean":
            return profiles.mean(axis=0)
        elif method == "sum":
            return profiles.sum(axis=0)
        elif method == "median":
            return np.median(profiles, axis=0)
        elif method in self._custom_aggregation:
            return self._custom_aggregation[method](profiles)
        else:
            raise ValueError(f"Unknown aggregation method {method}")



class GR_AggregatedProfilesUpdatable(RecAlgoGroupAggregated):
    def __init__(self, single_user_recommender: RecAlgoFull, custom_aggregation=None,
                update_mode: str = "ema", alpha: float = 0.3):
        """
        update_mode:
            'ema'  -> profile = (1 - alpha) * profile + alpha * vote_vec
            'mean' -> per-index running mean of votes; combined with the base profile
        alpha: EMA coefficient (0..1)
        """

        self._recommender: RecAlgoFull = single_user_recommender
        self._custom_aggregation = custom_aggregation or {}
        self._profiles: Dict[Tuple[int,...], np.ndarray] = {}     # group_key -> current profile
        self._vote_counts: Dict[Tuple[int,...], np.ndarray] = {}  # for 'mean' so we know how many users voted for given item
        self._update_mode = update_mode
        self._alpha = alpha

    # ---------- public API ----------
    def predict_group(self, user_ids: List[int], item_id: int, method: str = "mean") -> float:
        prof = self._ensure_profile(user_ids, method)
        scores = self._recommender.get_item_scores(prof)
        idx = self._recommender.item_id_to_index(item_id)
        return float(scores[idx])

    def top_k_iterator(self, user_ids: List[int], method: str = "mean", exclude: Set[int] | None = None) -> TopKIterator:
        prof = self._ensure_profile(user_ids, method)
        scores = self._recommender.get_item_scores(prof)
        pairs = [(self._recommender.index_to_item_id(i), float(s)) for i, s in enumerate(scores)]
        return TopKIterator(pairs, exclude=exclude)

    def update_group_with_votes(
        self,
        user_item_interactions: dict[int, dict[int, float]],
        *,
        reduce: str = "mean",   # "mean" | "sum"
    ) -> None:

        key = self._group_key_from_interactions(user_item_interactions)
        # 1. group key + init profile
        if key not in self._profiles:
            self._profiles[key] = self._aggregate_profiles(list(key), method="mean")

        # Initialise vote counters for 'mean' even when the profile already existed.
        if self._update_mode == "mean" and key not in self._vote_counts:
            self._vote_counts[key] = np.zeros_like(self._profiles[key])

        prof = self._profiles[key]
        # 2. reduce group last votes
        v_sum = np.zeros_like(prof)
        v_cnt = np.zeros_like(prof)

        for uid, items in user_item_interactions.items():
            for item_id, val in items.items():
                idx = self._recommender.item_id_to_index(item_id)
                v_sum[idx] += float(val)
                v_cnt[idx] += 1.0

        mask = v_cnt > 0
        if not np.any(mask):
            return

        v = np.zeros_like(prof)
        if reduce == "mean":
            v[mask] = v_sum[mask] / v_cnt[mask]
        elif reduce == "sum":
            v = v_sum
        else:
            raise ValueError(f"Unknown reduce='{reduce}'")

        # 3. update profile
        if self._update_mode == "ema":
            a = self._alpha
            self._profiles[key][mask] = (1 - a) * self._profiles[key][mask] + a * v[mask]
        elif self._update_mode == "mean":
            counts = self._vote_counts[key]
            old_counts = counts[mask]
            new_counts = old_counts + v_cnt[mask]
            self._profiles[key][mask] = (
                self._profiles[key][mask] * old_counts + v_sum[mask]
            ) / new_counts
            counts[mask] = new_counts
        else:
            raise ValueError(f"Unknown update_mode: {self._update_mode}")

    def recommend_group_top_k(
        self,
        user_ids: List[int],
        k: int,
        method: str = "mean",
        exclude: Set[int] | None = None,
        return_scores: bool = False,
    ) -> List[int]:

        if k == 0:
            return []

        prof = self._ensure_profile(user_ids, method)
        scores = self._recommender.get_item_scores(prof)

        n = len(scores)
        mask = np.ones(n, dtype=bool)
        if exclude:
            ex = {self._recommender.item_id_to_index(i) for i in exclude}
            if ex:
                mask[list(ex)] = False

        if not mask.any():
            return []

        k = min(k, int(mask.sum()))
        masked = np.full(n, -np.inf)
        masked[mask] = scores[mask]

        # Fast top-k without sorting the full vector.
        top_idx = np.argpartition(masked, -k)[-k:]
        top_idx = top_idx[np.argsort(masked[top_idx])[::-1]]

        if return_scores:
            return [(self._recommender.index_to_item_id(i), float(scores[i])) for i in top_idx]
        else:
            return [self._recommender.index_to_item_id(i) for i in top_idx]

    def reset_group_state(self, user_ids: List[int]) -> None:
        key = self._group_key(user_ids)
        self._profiles.pop(key, None)
        self._vote_counts.pop(key, None)

    # ---------- internal helpers ----------
    def _group_key(self, user_ids: List[int]) -> Tuple[int, ...]:
        return tuple(sorted(user_ids))

    def _group_key_from_interactions(self, user_item_interactions: dict[int, dict[int, float]]) -> tuple[int, ...]:
        """ Returns user ids in the group that provided interactions."""
        return tuple(sorted(user_item_interactions.keys()))


    def _ensure_profile(self, user_ids: List[int], method: str) -> np.ndarray: # group profile
        """ If group profile for given users does not exist =>
            creates new profile using provided method. """

        key = self._group_key(user_ids)
        if key not in self._profiles:
            self._profiles[key] = self._aggregate_profiles(user_ids, method)
        return self._profiles[key]

    def _aggregate_profiles(self, user_ids: List[int], method: str) -> np.ndarray:

        profs = [self._recommender.get_user_vector(uid) for uid in user_ids]
        profs = np.vstack(profs)

        if method == "mean":
            return profs.mean(axis=0)
        if method == "sum":
            return profs.sum(axis=0)
        if method == "median":
            return np.median(profs, axis=0)
        if method in ("multiplicative", "geomean"):
            eps = 1e-12
            X = np.clip(profs, eps, None)
            return np.exp(np.mean(np.log(X), axis=0))
        if method == "plurality":
            score_matrix = np.vstack([self._recommender.get_item_scores(p) for p in profs])
            top = np.argmax(score_matrix, axis=1)
            return np.bincount(top, minlength=score_matrix.shape[1]).astype(float)
        if method in self._custom_aggregation:
            return self._custom_aggregation[method](profs)
        raise ValueError(f"Unknown aggregation method {method}")