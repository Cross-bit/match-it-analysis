"""
`recommender_engine` — consensus recommendation engines used by mediators and evaluation pipelines.

This module contains:
- abstract interfaces for single-user and group recommendation engines,
- async engines (individual, group-aggregated, and updatable hybrid variants),
- sync engines (group-shared recommendations, with and without feedback updates),
- feedback-to-stars normalization helpers used by updatable pipelines,
- a commented usage sketch at the bottom (not executed).

An "engine" is an orchestration layer, not the scoring model itself.
The mediator drives the consensus loop (round progression, thresholds, and
redistribution), while the engine adapts recommender models to that loop.

In practice, the engine sits between:
- mediator logic (`consensus_mediator`) and
- recommender/scoring logic (individual/group recommenders and their iterators).

The engine is responsible for round-local recommendation state, for example:
- state reset between rounds/groups,
- exclusion tracking (do not recommend already served items),
- shared slate semantics (for group scenarios, all members in the same round
  receive the same generated group slate, instead of each call advancing the
  iterator independently),
- feedback updates (convert mediator vote format and apply updates to updatable
  recommender models when required by the strategy).

This separation keeps recommendation math inside recommender algorithms, while
consensus-specific orchestration remains reusable and consistent across evals.
"""

import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import numpy as np

from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoIterator
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import GR_AggregatedProfilesUpdatable, TopKIterator
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.models import Vote
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.interface import RecAlgoGroupAggregated, RecAlgoUpdatable


class GeneralRecommendationEngineBase(ABC):

    @abstractmethod
    def get_all_recommended_items(self) -> Set[int]:
        "Returns set of all the items recommended to the client during"
        pass

    @abstractmethod
    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """
        pass

    @abstractmethod
    def recommend_next_k(self, user_id: int, size: int, lastVotes: List[Vote] = None) -> List[int]:
        pass

class GroupRecommendationEngineBase(ABC):

    @abstractmethod
    def get_all_recommended_items(self) -> Set[int]:
        "Returns set of all the items recommended to the client during"
        pass

    @abstractmethod
    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """
        pass

    @abstractmethod
    def recommend_next_k(self, user_id: List[int], size: int, lastVotes: Dict[int, List[Vote]] = None) -> List[int]:
        pass


# ==========================================================
# Recommendation engines for async consensus algorithm
# ==========================================================



#
# Generates single-user recommendation to each member (the members data are not aggregated)
#
class RecommendationEngineIndividualEaser(GeneralRecommendationEngineBase):

    def __init__(self, users_ids: List[int], model_iterator: RecAlgoIterator):
        """ Each user gets his own single-user recommender recommendation.

        Args:
            user_ids (List[str]): Group user IDs.
            easer (EaserCached): Easer algorithm implementation.
        """

        self.model = model_iterator
        self.easer_iterators = None
        self.reset_iteration(users_ids)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
            Resets the recommender's internal state for a new recommendation session.

            Args:
                users_ids (List[int]): List of user IDs the recommender should consider.
                exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
                agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

            Note:
                This method should be called whenever the user group or configuration changes,
                to ensure the recommender reflects the new context.
        """
        self.easer_iterators = { user_id: self.model.top_k_iterator(user_id, exclude_items) for user_id in users_ids }
        self.all_recommended_item_ids = exclude_items

    def recommend_next_k(self, user_id: int, size: int, lastVotes: List[Vote] = None) -> List[int]:
        user_recommendation = [next(self.easer_iterators[user_id]) for _ in range(size)]
        return user_recommendation

#
# Generates different GROUP recommendation to each member (the members data are aggregated)
#
class RecommendationEngineGroupAllIndividualEaser(GeneralRecommendationEngineBase):

    def __init__(self, users_ids: List[int], model: RecAlgoGroupAggregated):
        self.all_recommended_item_ids = []
        self.model: RecAlgoGroupAggregated = model
        self.current_iterator = None
        self.users_ids = users_ids
        # One shared slate per mediator round (see begin_new_recommendation_round).
        self._group_sync_slate: Optional[List[int]] = None

        self.reset_iteration(users_ids)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def begin_new_recommendation_round(self) -> None:
        """Invalidate cached slate so the next recommend_next_k fills a fresh group list for this round."""
        self._group_sync_slate = None

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """

        if users_ids == []:
            return

        self.current_iterator = self.model.top_k_iterator(users_ids, agg_strategy, exclude_items)
        self.all_recommended_item_ids = exclude_items
        self.users_ids = users_ids
        self._group_sync_slate = None

    def recommend_next_k(self, user_id: int, size: int, last_votes: List[Vote] = None) -> List[int]:
        # Shared group slate: mediator calls recommend_next_k once per member per round.
        # Without caching, each call would advance the group iterator and members would see
        # disjoint items in the same round (consensus then becomes extremely unlikely for small W).
        if self._group_sync_slate is None or len(self._group_sync_slate) != size:
            self._group_sync_slate = [next(self.current_iterator) for _ in range(size)]
        self.last_recommendation_buffer = list(self._group_sync_slate)
        return self.last_recommendation_buffer


# For async hybrid with group recommender and feedback
class RecommendationEngineGroupAllIndividualEaserUpdatable(GeneralRecommendationEngineBase):

    def __init__(self, users_ids: List[int], updatable_group_model: GR_AggregatedProfilesUpdatable):
        self.all_recommended_item_ids = []
        self._all_recommended_item_ids_set: Set[int] = set()
        self.updatable_group_model: GR_AggregatedProfilesUpdatable = updatable_group_model
        self.current_iterator = None
        self.users_ids = users_ids

        self.served_users_in_current_round = 0
        self._group_sync_slate: Optional[List[int]] = None
        self.reset_iteration(users_ids)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def begin_new_recommendation_round(self) -> None:
        """
        Invalidate round-local recommendation cache.
        Must be called once at start of each mediator round.
        """
        self.served_users_in_current_round = 0
        self._group_sync_slate = None

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """

        if users_ids == []:
            return

        if exclude_items:
            self.all_recommended_item_ids = list(exclude_items)
            self._all_recommended_item_ids_set = set(exclude_items)
        else:
            self.all_recommended_item_ids = []
            self._all_recommended_item_ids_set = set()

        self.users_ids = users_ids
        self.begin_new_recommendation_round()

    def update_model(self, last_votes: Dict[int, List[Vote]]):
        if len(last_votes) > 0:
            mapped = self.normalize_feedback_to_stars(last_votes)
            self.updatable_group_model.update_group_with_votes(mapped, reduce="mean")

        # make sure we are really tracking all the items
        for votes in last_votes.values():
            for v in votes:
                if v.id not in self._all_recommended_item_ids_set:
                    self._all_recommended_item_ids_set.add(v.id)
                    self.all_recommended_item_ids.append(v.id)

    def recommend_next_k(self, user_id: int, size: int, last_votes: List[Vote] = None) -> List[int]:

        # UPDATING: we leave the update on the caller using -- update_model -- method
        if self._group_sync_slate is None or len(self._group_sync_slate) != size:
            self._group_sync_slate = self.updatable_group_model.recommend_group_top_k(
                self.users_ids,
                size,
                exclude=self._all_recommended_item_ids_set,
            )
            for item_id in self._group_sync_slate:
                if item_id not in self._all_recommended_item_ids_set:
                    self._all_recommended_item_ids_set.add(item_id)
                    self.all_recommended_item_ids.append(item_id)

        self.served_users_in_current_round += 1
        return list(self._group_sync_slate)

    def normalize_feedback_to_stars(
        self,
        user_votes: Dict[int, List["Vote"]],
        *,
        star_min: float = 1.0,
        star_max: float = 5.0,
    ) -> Dict[int, Dict[int, float]]:
        """
        Convert per-user lists of Vote objects into a nested dict with star ratings.

        Args:
            user_votes: { user_id: [Vote(item_id, value in {-1,0,1}), ...], ... }
            star_min:   value for -1 feedback (default 1.0)
            star_max:   value for +1 feedback (default 5.0)

        Returns:
            { user_id: { item_id: mapped_star (float) }, ... }
            Mapping uses midpoint for zero.
        """
        midpoint = 0.5 * (star_min + star_max)
        out: Dict[int, Dict[int, float]] = {}
        for uid, votes in user_votes.items():
            mapped: Dict[int, float] = {}
            for v in votes:
                val = float(getattr(v, "value"))
                iid = int(getattr(v, "id"))
                if val > 0:
                    mapped[iid] = float(star_max)
                elif val < 0:
                    mapped[iid] = float(star_min)
                else:  # val == 0
                    mapped[iid] = float(midpoint)
            if mapped:  # skip empty
                out[uid] = mapped

        return out


class RecommendationEngineSTSGroupDynamic(GeneralRecommendationEngineBase):
    """
    STSGroup-inspired dynamic engine with:
    - individual utility profiles updated from vote constraints,
    - weighted group profile aggregation,
    - one shared group recommendation slate per round.
    """

    _EPS = 1e-12

    def __init__(
        self,
        users_ids: List[int],
        model_iterator: RecAlgoIterator,
        *,
        beta: float = 0.6,
        learning_rate: float = 0.05,
        margin: float = 1e-3,
        max_constraint_updates: int = 64,
    ):
        self.model = model_iterator
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        self.margin = float(margin)
        self.max_constraint_updates = int(max_constraint_updates)

        b_mat = getattr(self.model, "B", None)
        self._item_embeddings = (
            np.asarray(b_mat.T, dtype=np.float64)
            if b_mat is not None
            else None
        )

        self.users_ids: List[int] = []
        self._base_profiles: Dict[int, np.ndarray] = {}
        self._current_profiles: Dict[int, np.ndarray] = {}
        self._action_counts: Dict[int, float] = {}
        self._group_profile: Optional[np.ndarray] = None
        self._group_sync_slate: Optional[List[int]] = None
        self._all_recommended_items_set: Set[int] = set()
        self._all_recommended_item_ids: List[int] = []

        self.reset_iteration(users_ids)
        super().__init__()

    def _normalize_profile(self, vec: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(vec, dtype=np.float64), 0.0, None)
        s = float(arr.sum())
        if s <= self._EPS:
            n = max(1, int(arr.shape[0]))
            return np.full(n, 1.0 / float(n), dtype=np.float64)
        return arr / s

    def _item_score_for_profile(self, profile: np.ndarray, item_idx: int) -> float:
        if self._item_embeddings is None:
            return float(self.model.get_item_scores(profile)[item_idx])
        return float(np.dot(profile, self._item_embeddings[item_idx]))

    def _compute_weighted_group_profile(self) -> np.ndarray:
        total = float(sum(self._action_counts.values()))
        if total <= self._EPS:
            mat = np.vstack([self._current_profiles[uid] for uid in self.users_ids])
            return self._normalize_profile(mat.mean(axis=0))
        acc = None
        for uid in self.users_ids:
            alpha = float(self._action_counts.get(uid, 0.0)) / total
            prof = self._current_profiles[uid]
            acc = alpha * prof if acc is None else acc + alpha * prof
        return self._normalize_profile(acc)

    def begin_new_recommendation_round(self) -> None:
        self._group_sync_slate = None

    def get_all_recommended_items(self) -> Set[int]:
        return set(self._all_recommended_items_set)

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        self.users_ids = list(users_ids)
        self._base_profiles = {}
        self._current_profiles = {}
        self._action_counts = {}
        for uid in self.users_ids:
            p = self._normalize_profile(self.model.get_user_vector(uid))
            self._base_profiles[uid] = p
            self._current_profiles[uid] = p.copy()
            self._action_counts[uid] = 1.0
        self._group_profile = self._compute_weighted_group_profile()
        self._group_sync_slate = None
        self._all_recommended_items_set = set(exclude_items or set())
        self._all_recommended_item_ids = list(exclude_items or [])

    def get_individual_item_score(self, user_id: int, item_id: int) -> float:
        item_idx = int(self.model.item_id_to_index(item_id))
        return self._item_score_for_profile(self._current_profiles[user_id], item_idx)

    def _enforce_pairwise_order(self, profile: np.ndarray, high_ids: List[int], low_ids: List[int]) -> np.ndarray:
        if not high_ids or not low_ids:
            return profile
        p = profile.copy()
        updates = 0
        for hi in high_ids:
            for lo in low_ids:
                if updates >= self.max_constraint_updates:
                    return self._normalize_profile(p)
                if self._item_score_for_profile(p, hi) <= self._item_score_for_profile(p, lo) + self.margin:
                    if self._item_embeddings is None:
                        delta = np.zeros_like(p)
                        delta[hi] += 1.0
                        delta[lo] -= 1.0
                    else:
                        delta = self._item_embeddings[hi] - self._item_embeddings[lo]
                    p = self._normalize_profile(p + self.learning_rate * delta)
                    updates += 1
        return self._normalize_profile(p)

    def _solve_group_conditioned_profile(self, liked: List[int], neutral: List[int], disliked: List[int]) -> np.ndarray:
        cand = self._group_profile.copy()
        cand = self._enforce_pairwise_order(cand, liked, neutral)
        cand = self._enforce_pairwise_order(cand, neutral, disliked)
        cand = self._enforce_pairwise_order(cand, liked, disliked)
        return self._normalize_profile(cand)

    def update_model(self, last_votes: Dict[int, List[Vote]]) -> None:
        if not last_votes:
            return
        for uid, votes in last_votes.items():
            liked: List[int] = []
            neutral: List[int] = []
            disliked: List[int] = []
            for v in votes:
                idx = int(self.model.item_id_to_index(int(v.id)))
                val = float(v.value)
                if val > 0:
                    liked.append(idx)
                elif val < 0:
                    disliked.append(idx)
                else:
                    neutral.append(idx)
            if not liked and not neutral and not disliked:
                continue
            self._action_counts[uid] = float(self._action_counts.get(uid, 1.0) + len(votes))
            inferred = self._solve_group_conditioned_profile(liked, neutral, disliked)
            old = self._current_profiles[uid]
            self._current_profiles[uid] = self._normalize_profile(self.beta * old + (1.0 - self.beta) * inferred)
        self._group_profile = self._compute_weighted_group_profile()

    def _top_items_from_group_profile(self, k: int) -> List[int]:
        if k <= 0:
            return []
        scores = self.model.get_item_scores(self._group_profile)
        n = len(scores)
        mask = np.ones(n, dtype=bool)
        if self._all_recommended_items_set:
            ex_idx = [self.model.item_id_to_index(iid) for iid in self._all_recommended_items_set]
            if ex_idx:
                mask[ex_idx] = False
        if not mask.any():
            return []
        k = min(k, int(mask.sum()))
        masked = np.full(n, -np.inf, dtype=np.float64)
        masked[mask] = scores[mask]
        top_idx = np.argpartition(masked, -k)[-k:]
        top_idx = top_idx[np.argsort(masked[top_idx])[::-1]]
        top_items = [int(self.model.index_to_item_id(int(i))) for i in top_idx]
        for iid in top_items:
            if iid not in self._all_recommended_items_set:
                self._all_recommended_items_set.add(iid)
                self._all_recommended_item_ids.append(iid)
        return top_items

    def recommend_next_k(self, user_id: int, size: int, lastVotes: List[Vote] = None) -> List[int]:
        if self._group_sync_slate is None or len(self._group_sync_slate) != size:
            self._group_sync_slate = self._top_items_from_group_profile(size)
        return list(self._group_sync_slate)



### =============================================================================================================


# ==========================================================
# Recommendation engines for sync consensus algorithm
# ==========================================================

#
# Generates same recommendation to each member (the members data are aggregated)
#
class RecommendationEngineGroupAllSameEaser(GroupRecommendationEngineBase):

    def __init__(self, users_ids: List[int], ratings_matrix: pd.DataFrame, model: RecAlgoGroupAggregated, default_strategy='mean'):
        self.ratings_matrix:pd.DataFrame = ratings_matrix
        self.all_recommended_item_ids = []
        self.model: RecAlgoGroupAggregated = model
        self.current_iterator = None
        self.users_ids = users_ids
        self.last_recommendation_buffer = [] # stores last group recommendation
        self.number_of_users_recommended_to_in_round = 0
        self.users_served_in_this_round = set(users_ids) # all the users we served in current round

        self.reset_iteration(users_ids, agg_strategy=default_strategy)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """

        self.current_iterator = self.model.top_k_iterator(users_ids, agg_strategy, exclude_items)
        self.all_recommended_item_ids = exclude_items
        self.users_ids = users_ids
        self.last_recommendation_buffer = [] # stores last group recommendation
        self.users_served_in_this_round = set(users_ids)

    def recommend_next_k(self, user_id: List[int], size: int, lastVotes: Dict[int, List[Vote]] = None) -> List[int]:
        self.last_recommendation_buffer = [next(self.current_iterator) for _ in range(size)]
        return self.last_recommendation_buffer

    def _all_group_members_served(self) -> bool:
        return len(self.users_served_in_this_round) == len(self.users_ids)

    def _update_served_users(self, user_id: int):
        self.users_served_in_this_round.add(user_id)


class RecommendationEngineGroupAllSameEaserWithFeedback(GroupRecommendationEngineBase):

    def __init__(self, users_ids: List[int], model: GR_AggregatedProfilesUpdatable, default_strategy='mean'):
        self.all_recommended_item_ids = []
        self.model: GR_AggregatedProfilesUpdatable = model
        self.current_iterator = None
        self.users_ids = users_ids
        self.last_recommendation_buffer = [] # stores last group recommendation
        self.number_of_users_recommended_to_in_round = 0
        self.users_served_in_this_round = set(users_ids) # all the users we served in current round

        self.reset_iteration(users_ids, agg_strategy=default_strategy)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        pass

    def recommend_next_k(self, user_ids: List[int], size: int, lastVotes: Dict[int, List[Vote]] = None) -> List[int]:

        if len(self.all_recommended_item_ids) > 0:
            mapped = self.normalize_feedback_to_stars(lastVotes) # {-1,0,1} -> 1..5 (0 -> midpoint)
            self.model.update_group_with_votes(mapped, reduce="mean")

        new_rec = self.model.recommend_group_top_k(user_ids, size, exclude=set(self.all_recommended_item_ids))
        self.all_recommended_item_ids.extend(new_rec)

        return new_rec

    def normalize_feedback_to_stars(
        self,
        user_votes: Dict[int, List["Vote"]],
        *,
        star_min: float = 1.0,
        star_max: float = 5.0,
    ) -> Dict[int, Dict[int, float]]:
        """
        Convert per-user lists of Vote objects into a nested dict with star ratings.

        Args:
            user_votes: { user_id: [Vote(item_id, value in {-1,0,1}), ...], ... }
            star_min:   value for -1 feedback (default 1.0)
            star_max:   value for +1 feedback (default 5.0)

        Returns:
            { user_id: { item_id: mapped_star (float) }, ... }
            Mapping uses midpoint for zero.
        """
        midpoint = 0.5 * (star_min + star_max)
        out: Dict[int, Dict[int, float]] = {}
        for uid, votes in user_votes.items():
            mapped: Dict[int, float] = {}
            for v in votes:
                val = float(getattr(v, "value"))
                iid = int(getattr(v, "id"))
                if val > 0:
                    mapped[iid] = float(star_max)
                elif val < 0:
                    mapped[iid] = float(star_min)
                else:
                    mapped[iid] = float(midpoint)
            if mapped:  # skip empty
                out[uid] = mapped

        return out


# --- Usage sketch (commented): same group recommendation for all members ---
# Not executed when this file is imported. Copy into a script or REPL if needed.
#
# from dataset.data_access import MovieLensDatasetLoader
# from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import (
#     GR_AggregatedRecommendations,
# )
#
# d_loader = MovieLensDatasetLoader()
# _, ratings_matrix = d_loader.load_data(True)
# group_model = GR_AggregatedRecommendations()
# group_model.fit(ratings_matrix)
# user_ids = [42, 24, 5, 6]
# rec_engine = RecommendationEngineGroupAllSameEaser(user_ids, ratings_matrix, group_model)
# rec_engine.recommend_next_k(42, 5)  # same slate for other members in that round
# rec_engine.recommend_next_k(24, 5)
# rec_engine.reset_iteration([42, 24], agg_strategy="median")

