"""
`consensus_mediator` — round-by-round orchestration for group consensus simulations.

Each **mediator** turns last-round ``Vote`` objects into the next per-user list of
candidate ``item_id`` values inside a fixed interaction window ``window_size``.
Implementations differ in how much is **new recommendation** vs **re-offered**
(redistributed) content, and whether the slate is **shared** across the group.

**Composed pieces**

- **Recommendation engines** (``recommender_engine``): per-user or group engines
  that wrap underlying recommenders / iterators.
- **``RedistributionUnit``** (async / hybrid tails): priority queues that recycle
  positively liked items users have not resolved yet.
- **``ThresholdPolicy``**: decides how many of the ``window_size`` slots are taken
  from redistribution vs. fresh recommendation each round (after round 0).

**Threshold policies**

- ``ThresholdPolicyStatic`` — round ``0`` allocates **no** redistribution
  (``t = 0`` so the full window is fresh items); every later round uses a fixed
  integer ``t`` (still clamped to the user's redistribution queue length in the
  mediator).
- ``ThresholdPolicySigmoid`` — after round ``0``, computes a **continuous** split
  factor from (a) a shifted logistic in the **round index** and (b) a **queue
  filling** scaler read from ``RedistributionContext``; multiplies by
  ``window_size`` and rounds to an integer slot count for redistribution.

**Mediator variants**

- ``ConsensusMediatorAsyncApproach`` — classic **async** split: update queues,
  draw ``t`` redistributed items, fill remainder with the general recommender.
- ``ConsensusMediatorSyncApproach`` — **sync**: one group recommendation vector per
  round, broadcast to every member; no redistribution.
- ``ConsensusMediatorHybridApproach`` — **hybrid preamble**: first delivers
  ``first_round_ration`` total **group-sync** items across early rounds (mixed with
  per-user tail inside each round when ``W`` allows), then enters async-style
  redistribution + per-user recommendations (uses a separate internal async round
  counter for the sigmoid policy).
- ``ConsensusMediatorHybridApproachWithFeedback`` — like hybrid, but the async
  phase uses an **updatable** per-user engine that can ingest votes via
  ``update_model``; redistribution bootstraps from votes with unanimous overlap
  filtered out on first entry.

**Shared helpers**

``check_matches`` / ``clear_item_votes`` support RFC-style diagnostics (unanimous
positive votes on an item across the whole group).
"""

from collections import defaultdict
import random
import numpy as np
from typing import List, Dict, Set, Literal

from abc import ABC, abstractmethod
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import GeneralRecommendationEngineBase, GroupRecommendationEngineBase, RecommendationEngineGroupAllIndividualEaserUpdatable, RecommendationEngineGroupAllSameEaser, RecommendationEngineGroupAllSameEaserWithFeedback, RecommendationEngineIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionContext, RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.models import Vote
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer_cached import EaserCached
from dataset.data_access import MovieLensDatasetLoader

class ThresholdPolicy(ABC):
    """How many redistribution slots ``t`` to take before filling the rest with new items."""

    @abstractmethod
    def get_parameter_value(self, round: int, user_id: int):
        pass

    def get_metadata(self):
        return {
            "type": self.__class__.__name__,
            "general_desc": "Threshold parameter policy."
        }

class ThresholdPolicyStatic(ThresholdPolicy):
    """Constant ``t`` after the opening round; ``t = 0`` on round ``0``."""

    def __init__(self, t_param: int):
        self.t = t_param
        super().__init__()

    def get_parameter_value(self, round: int, user_id: int):
        if (round == 0):
            return 0
        else:
            return self.t

    def get_metadata(self):
        base = super().get_metadata()
        base["t_param"] = self.t
        return base

class ThresholdPolicySigmoid(ThresholdPolicy):
    """Smooth ``t`` schedule: logistic over rounds × queue-fill scaler × ``window_size``."""

    def __init__(self, red_context: RedistributionContext, window_size, sigmoid_center = 5, sigmoid_steepness = 1.4, c_init: float = 0.2, max_filling = 10, min_filling=0):
        """
        Args:
            red_context: Must expose per-user redistribution queue sizes (``RedistributionUnit``).
            window_size: Interaction window ``W``; returned ``t`` is capped later by queue length.
            sigmoid_center: Inflection round for the logistic term.
            sigmoid_steepness: Logistic slope ``a``.
            c_init: Lower asymptote ``c`` inside ``(0, 1)`` — minimum fraction of ``W`` devoted
                to redistribution even at early post-round-0 rounds (before scaling effects).
            max_filling / min_filling: Bounds passed to the queue-fill scaler (redistribution depth signal).
        """
        self.redistribution_context = red_context
        self.min_filling = min_filling  # lower bound for queue-fill normalization
        self.max_filling = max_filling
        self.c_init = c_init
        self.steepness = sigmoid_steepness
        epsilon = 0.015  # tolerance used when solving for the transition round offset
        self.sigmoid_center = sigmoid_center
        self.transition_point = self.get_transition_point(sigmoid_center, sigmoid_steepness, c_init, epsilon)
        self.window_size = window_size
        super().__init__()

    def scaler(self, filling, min_filling = 20, max_filling = 70):
        return np.maximum(0, np.minimum(filling, max_filling) - min_filling)/(max_filling - min_filling)

    def modified_sigmoid_with_upper_bound(self, x, c, k0, a, filling, min_filling, max_filling):
        return c + ((1 - c) / (1 + np.exp(-a * (x - k0)))) * self.scaler(filling, min_filling, max_filling)

    def get_transition_point(self, sigmoid_center_x, alpha, c_0, epsilon):
        """ the x value where value of sigmoid is less than c_0 with epsilon accuracy interval, alpha -- steepness of the sigmoid"""
        delta_epsilon = (1 / alpha) * np.log((1 - c_0) / epsilon - 1)
        x_constant_ends = sigmoid_center_x - delta_epsilon
        return x_constant_ends

    def get_parameter_value(self, current_round: int, user_id: int):
        if (current_round == 0):
            return 0
        else:
            queue_filling = self.redistribution_context.get_user_queue_size(user_id)
            value = self.modified_sigmoid_with_upper_bound(current_round, self.c_init, self.transition_point, self.steepness, queue_filling, self.min_filling, self.max_filling)
            return int(round(float(self.window_size * value)))


    def get_metadata(self):
        base = super().get_metadata()
        base.update({
            "sigmoid_center": self.sigmoid_center,
            "sigmoid_steepness": self.steepness,
            "c_init": self.c_init,
            "max_filling": self.max_filling,
            "min_filling": self.min_filling,
            "transition_point": self.transition_point,
        })
        return base

class ConsensusMediatorBase(ABC):
    """Abstract consensus driver shared by async / sync / hybrid mediators."""

    @abstractmethod
    def get_next_round_recommendation(self, previous_round_votes: Dict[int, List[Vote]]) -> Dict[int, List[int]]:
        pass

    @abstractmethod
    def clear_item_votes(self, item_id: int):
        pass

    @abstractmethod
    def check_matches(self, previous_round_votes: Dict[int, List[Vote]]) -> List[int]:
        pass


class ConsensusMediatorAsyncApproach(ConsensusMediatorBase):
    """Async mediator: per-user window mix of fresh recommendations + redistributed items."""

    def __init__(
                self,
                users_ids: List[int],
                recommender: GeneralRecommendationEngineBase,
                redistribution_unit: RedistributionUnit,
                threshold_policy: ThresholdPolicy,
                window_size: int
                ):
        self.general_recommender = recommender
        self.redistribution_unit = redistribution_unit
        self.threshold_policy = threshold_policy
        self.current_round = 0
        self.window_size = window_size
        self.users_ids = users_ids
        self.users_positive_votes: Dict[int, Set[int]] = { user_id: set() for user_id in users_ids } # mapping of user to his positive votes over entire voting session
        self.all_recommended_items_votes = defaultdict(int) # list of all recommended/redistributed items so far (all items that appeared)

        if (len(users_ids) != len(set(users_ids))):
            raise ValueError("Duplicate user UUIDs found!")

    def update_on_group_size_changed(self, new_users_ids: List[int]) -> None:
        self.users_ids = new_users_ids
        self.general_recommender.reset_iteration(new_users_ids, set(self.all_recommended_items_votes.keys()))

    def get_next_round_recommendation(self, previous_round_votes: Dict[int, List[Vote]]) -> Dict[int, List[int]]:
        """ Updates the internal state based on user votes from previous rounds
            and returns a new list of recommended items for each user.
        Args:
            users_votes (Dict[str, List[Item]]): Users votes from the last round.
        Returns:
            List[int]: Recommendation for next round.
        """

        next_round_recs: Dict[int, List[Vote]] = {}

        _begin = getattr(self.general_recommender, "begin_new_recommendation_round", None)
        if callable(_begin):
            _begin()

        ## 1. Update redistribution Unit
        self.redistribution_unit.update_voted_items(previous_round_votes)

        for user_id, votes in previous_round_votes.items():

            # 2. Threshold t: how many slots come from redistribution (bounded by queue size).
            redistribution_queue_size = self.redistribution_unit.get_user_redistribution_queue_size(user_id)
            redistributed_part_size = min(
                self.threshold_policy.get_parameter_value(self.current_round, user_id),
                redistribution_queue_size,
                self.window_size,  # never exceed W slots per round
            )
            redistributed_part_size = int(round(float(redistributed_part_size)))
            new_recommendation_size = self.window_size - redistributed_part_size

            # 3. Recommend new items
            new_recommendations = self.general_recommender.recommend_next_k(user_id, new_recommendation_size, votes)
            next_round_recs[user_id] = new_recommendations

            # 4. Redistribute items from previous rounds
            redistributed_items = self.redistribution_unit.get_redistributed_items(user_id, redistributed_part_size)
            next_round_recs[user_id] = next_round_recs[user_id] + redistributed_items

        self.current_round += 1

        return next_round_recs

    def _update_all_recommended_items(self, last_recommendation: Dict[int, List[int]]) -> None:
        for _, items in last_recommendation.items():
            for item_id in items:
                if item_id not in self.all_recommended_items_votes:
                    self.all_recommended_items_votes[item_id] = 0 # if it is new item

    def clear_item_votes(self, item_id: int):
        self.all_recommended_items_votes[item_id] = 0

    def check_matches(self, users_votes: Dict[int, List[Vote]]) -> List[int]:
        matched_items = []

        for user_id, votes in users_votes.items():
            for vote in votes:
                if vote.value == 1:
                    self.users_positive_votes[user_id].add(vote)
                    self.all_recommended_items_votes[vote.id] += 1

        for item_id, votes_count in self.all_recommended_items_votes.items():
            if votes_count == len(users_votes):
                matched_items.append(item_id)

        return matched_items


class ConsensusMediatorSTSGroupDynamic(ConsensusMediatorAsyncApproach):
    """
    STSGroup-style mediator variant.

    Behaviour is async redistribution with threshold policy, but it explicitly
    updates a dynamic STS group engine (if present) from previous votes.
    """

    def get_next_round_recommendation(self, previous_round_votes: Dict[int, List[Vote]]) -> Dict[int, List[int]]:
        updater = getattr(self.general_recommender, "update_model", None)
        if callable(updater) and previous_round_votes:
            updater(previous_round_votes)
        return super().get_next_round_recommendation(previous_round_votes)


class ConsensusMediatorSyncApproach(ConsensusMediatorBase):
    """Sync mediator: identical group slate for every member each round; no redistribution."""

    def __init__(self, users_ids: List[int], window_size: int, group_recommendation_engine: GroupRecommendationEngineBase):
        self.current_round = 0
        self.users_ids = users_ids
        self.window_size = window_size
        self.general_recommender = group_recommendation_engine
        self.all_recommended_items_votes = defaultdict(int) # list of all recommended/redistributed items so far (all items that appeared)

        if (len(users_ids) != len(set(users_ids))):
            raise ValueError("Duplicate user UUIDs found!")

    def update_on_group_size_changed(self, new_users_ids: List[int]) -> None:
        self.users_ids = new_users_ids
        self.general_recommender.reset_iteration(new_users_ids, set(self.all_recommended_items_votes.keys()))

    def get_next_round_recommendation(self, previous_round_votes: Dict[int, List[Vote]]) -> Dict[int, List[int]]:
        """ Updates the internal state based on user votes from previous rounds
            and returns a new list of recommended items for each user.

            Args:
                users_votes (Dict[str, List[Item]]): Users votes from the last round.
            Returns:
                List[int]: Recommendation for next round.
        """

        new_group_recommendation = self.general_recommender.recommend_next_k(previous_round_votes.keys(), self.window_size, previous_round_votes)
        next_round_recs: Dict[int, List[Vote]] = {}

        # For each user get new recommendation
        for user_id, votes in previous_round_votes.items():
            next_round_recs[user_id] = new_group_recommendation

        self.current_round += 1

        return next_round_recs

    def clear_item_votes(self, item_id: int):
        self.all_recommended_items_votes[item_id] = 0

    def check_matches(self, users_votes: Dict[int, List[Vote]]) -> List[int]:
        matched_items = []

        for user_id, votes in users_votes.items():
            for vote in votes:
                if vote.value == 1:
                    self.all_recommended_items_votes[vote.id] += 1

        for item_id, votes_count in self.all_recommended_items_votes.items():
            if votes_count == len(users_votes):
                matched_items.append(item_id)

        return matched_items


class ConsensusMediatorHybridApproach(ConsensusMediatorBase):
    """Hybrid mediator: optional sync preamble, then async redistribution + per-user fill."""

    def __init__(self, users_ids: List[int], general_recommender: GeneralRecommendationEngineBase, group_recommendation_engine: GroupRecommendationEngineBase, redistribution_unit: RedistributionUnit, first_round_ration: int, threshold_policy: ThresholdPolicy, window_size: int):
        self.general_recommender = general_recommender
        self.group_recommendation_engine = group_recommendation_engine

        self.redistribution_unit = redistribution_unit
        self.threshold_policy = threshold_policy

        if first_round_ration < 1:
            raise ValueError(
                f"Invalid first_round_ration: must be >= 1 (total group-sync items in opening phase). Got {first_round_ration}"
            )

        self.first_round_ration = first_round_ration
        self._sync_preamble_target = first_round_ration
        self._sync_preamble_delivered = 0
        self._hybrid_async_entered = False
        self._async_policy_round = 0

        self.current_round = 0
        self.window_size = window_size
        self.users_ids = users_ids
        self.users_positive_votes: Dict[int, Set[int]] = { user_id: set() for user_id in users_ids } # mapping of user to his positive votes over entire voting session
        self.all_recommended_items_votes = defaultdict(int) # list of all recommended/redistributed items so far (all items that appeared)

        if (len(users_ids) != len(set(users_ids))):
            raise ValueError("Duplicate user UUIDs found!")

    def update_on_group_size_changed(self, new_users_ids: List[int]) -> None:
        self.users_ids = new_users_ids
        self.general_recommender.reset_iteration(new_users_ids, set(self.all_recommended_items_votes.keys()))

    def get_next_round_recommendation(self, previous_round_votes: Dict[int, List[Vote]]) -> Dict[int, List[int]]:
        """ Updates the internal state based on user votes from previous rounds
            and returns a new list of recommended items for each user.
        Args:
            users_votes (Dict[str, List[Item]]): Users votes from the last round.
        Returns:
            List[int]: Recommendation for next round.
        """

        next_round_recs: Dict[int, List[int]] = {}

        if self._sync_preamble_delivered < self._sync_preamble_target:
            need = self._sync_preamble_target - self._sync_preamble_delivered
            take = min(self.window_size, need)
            new_sync_recommendations = self.group_recommendation_engine.recommend_next_k(self.users_ids, take, None)
            for user_id in self.users_ids:
                next_round_recs[user_id] = list(new_sync_recommendations)
            self._sync_preamble_delivered += take

            remaining_recommendation_size = self.window_size - take
            if remaining_recommendation_size > 0:
                exclude_so_far: Set[int] = set()
                for items in next_round_recs.values():
                    exclude_so_far.update(items)
                self.general_recommender.reset_iteration(users_ids=self.users_ids, exclude_items=exclude_so_far)
                for user_id in self.users_ids:
                    async_part = self.general_recommender.recommend_next_k(
                        user_id, remaining_recommendation_size, None
                    )
                    next_round_recs[user_id] = next_round_recs[user_id] + async_part

            self._update_all_recommended_items(next_round_recs)
            self.current_round += 1
            return next_round_recs

        if not self._hybrid_async_entered:
            self._hybrid_async_entered = True
            self.general_recommender.reset_iteration(
                users_ids=self.users_ids, exclude_items=set(self.all_recommended_items_votes.keys())
            )
            per_user_unique_votes = self._filter_all_same_items_out(previous_round_votes)
            self.redistribution_unit.update_voted_items(per_user_unique_votes)
        else:
            self.redistribution_unit.update_voted_items(previous_round_votes)

        self._async_policy_round += 1
        for user_id, votes in previous_round_votes.items():
            redistribution_queue_size = self.redistribution_unit.get_user_redistribution_queue_size(user_id)
            redistributed_part_size = min(
                self.threshold_policy.get_parameter_value(self._async_policy_round, user_id),
                redistribution_queue_size,
                self.window_size,  # keep round output bounded by W
            )
            redistributed_part_size = int(round(float(redistributed_part_size)))
            new_recommendation_size = self.window_size - redistributed_part_size

            new_sync_recommendations = self.general_recommender.recommend_next_k(user_id, new_recommendation_size, votes)
            next_round_recs[user_id] = new_sync_recommendations

            redistributed_items = self.redistribution_unit.get_redistributed_items(user_id, redistributed_part_size)
            next_round_recs[user_id] = next_round_recs[user_id] + redistributed_items

        self.current_round += 1
        return next_round_recs

    def _filter_all_same_items_out(self, previous_round_votes: Dict[int, List[Vote]]):

        sets = [set(votes) for votes in previous_round_votes.values()]
        common_votes = set.intersection(*sets)

        cleaned: Dict[int, List[Vote]] = {
            user_id: [v for v in votes if v not in common_votes]
            for user_id, votes in previous_round_votes.items()
        }

        return cleaned


    def _update_all_recommended_items(self, last_recommendation: Dict[int, List[int]]) -> None:
        for user_id, items in last_recommendation.items():
            for item_id in items:
                if item_id not in self.all_recommended_items_votes:
                    self.all_recommended_items_votes[item_id] = 0 # if it is new item

    def clear_item_votes(self, item_id: int):
        self.all_recommended_items_votes[item_id] = 0

    def check_matches(self, users_votes: Dict[int, List[Vote]]) -> List[int]:
        matched_items = []

        for user_id, votes in users_votes.items():
            for vote in votes:
                if vote.value == 1:
                    self.users_positive_votes[user_id].add(vote)
                    self.all_recommended_items_votes[vote.id] += 1

        for item_id, votes_count in self.all_recommended_items_votes.items():
            if votes_count == len(users_votes):
                matched_items.append(item_id)

        return matched_items


class ConsensusMediatorHybridApproachWithFeedback(ConsensusMediatorBase):
    """Hybrid mediator with vote-driven updates to the per-user engine in the async phase."""

    def __init__(self, users_ids: List[int],
                updatable_group_recommender: RecommendationEngineGroupAllIndividualEaserUpdatable,
                group_recommendation_engine: GroupRecommendationEngineBase,
                redistribution_unit: RedistributionUnit, first_round_ration: int, threshold_policy: ThresholdPolicy, window_size: int):

        self.updatable_group_recommender = updatable_group_recommender
        self.group_recommendation_engine = group_recommendation_engine

        self.redistribution_unit = redistribution_unit
        self.threshold_policy = threshold_policy

        if first_round_ration < 1:
            raise ValueError(
                f"Invalid first_round_ration: must be >= 1 (total group-sync items in opening phase). Got {first_round_ration}"
            )

        self.first_round_ration = first_round_ration
        self._sync_preamble_target = first_round_ration
        self._sync_preamble_delivered = 0
        self._hybrid_async_entered = False
        self._async_policy_round = 0

        self.current_round = 0
        self.window_size = window_size
        self.users_ids = users_ids
        self.users_positive_votes: Dict[int, Set[int]] = { user_id: set() for user_id in users_ids } # mapping of user to his positive votes over entire voting session
        self.all_recommended_items_votes = defaultdict(int) # list of all recommended/redistributed items so far (all items that appeared)

        if (len(users_ids) != len(set(users_ids))):
            raise ValueError("Duplicate user UUIDs found!")

    def update_on_group_size_changed(self, new_users_ids: List[int]) -> None:
        self.users_ids = new_users_ids
        self.updatable_group_recommender.reset_iteration(new_users_ids, set(self.all_recommended_items_votes.keys()))

    def get_next_round_recommendation(self, previous_round_votes: Dict[int, List[Vote]]) -> Dict[int, List[int]]:
        """ Updates the internal state based on user votes from previous rounds
            and returns a new list of recommended items for each user.
        Args:
            users_votes (Dict[str, List[Item]]): Users votes from the last round.
        Returns:
            List[int]: Recommendation for next round.
        """
        next_round_recs: Dict[int, List[int]] = {}
        _begin = getattr(self.updatable_group_recommender, "begin_new_recommendation_round", None)
        if callable(_begin):
            _begin()

        if self._sync_preamble_delivered < self._sync_preamble_target:
            need = self._sync_preamble_target - self._sync_preamble_delivered
            take = min(self.window_size, need)
            new_sync_recommendations = self.group_recommendation_engine.recommend_next_k(self.users_ids, take, None)
            for user_id in self.users_ids:
                next_round_recs[user_id] = list(new_sync_recommendations)
            self._sync_preamble_delivered += take

            remaining_recommendation_size = self.window_size - take
            if remaining_recommendation_size > 0:
                exclude_so_far: Set[int] = set()
                for items in next_round_recs.values():
                    exclude_so_far.update(items)
                self.updatable_group_recommender.reset_iteration(users_ids=self.users_ids, exclude_items=exclude_so_far)
                for user_id in self.users_ids:
                    async_part = self.updatable_group_recommender.recommend_next_k(
                        user_id, remaining_recommendation_size, None
                    )
                    next_round_recs[user_id] = next_round_recs[user_id] + async_part

            self._update_all_recommended_items(next_round_recs)
            self.current_round += 1
            return next_round_recs

        if not self._hybrid_async_entered:
            self._hybrid_async_entered = True
            per_user_unique_votes = self._filter_only_unique_items(previous_round_votes)
            self.redistribution_unit.update_voted_items(per_user_unique_votes)
        else:
            self.redistribution_unit.update_voted_items(previous_round_votes)

        self.updatable_group_recommender.update_model(previous_round_votes)

        self._async_policy_round += 1
        for user_id, votes in previous_round_votes.items():
            redistribution_queue_size = self.redistribution_unit.get_user_redistribution_queue_size(user_id)
            redistributed_part_size = min(
                self.threshold_policy.get_parameter_value(self._async_policy_round, user_id),
                redistribution_queue_size,
                self.window_size,  # keep round output bounded by W
            )
            redistributed_part_size = int(round(float(redistributed_part_size)))
            new_recommendation_size = self.window_size - redistributed_part_size

            new_sync_recommendations = self.updatable_group_recommender.recommend_next_k(user_id, new_recommendation_size, votes)
            next_round_recs[user_id] = new_sync_recommendations

            redistributed_items = self.redistribution_unit.get_redistributed_items(user_id, redistributed_part_size)
            next_round_recs[user_id] = next_round_recs[user_id] + redistributed_items

        self.current_round += 1
        return next_round_recs

    def _filter_only_unique_items(self, previous_round_votes: Dict[int, List[Vote]]):

        sets = [set(votes) for votes in previous_round_votes.values()]
        common_votes = set.intersection(*sets)

        cleaned: Dict[int, List[Vote]] = {
            user_id: [v for v in votes if v not in common_votes]
            for user_id, votes in previous_round_votes.items()
        }

        return cleaned

    def _update_all_recommended_items(self, last_recommendation: Dict[int, List[int]]) -> None:
        for user_id, items in last_recommendation.items():
            for item_id in items:
                if item_id not in self.all_recommended_items_votes:
                    self.all_recommended_items_votes[item_id] = 0 # if it is new item

    def clear_item_votes(self, item_id: int):
        self.all_recommended_items_votes[item_id] = 0

    def check_matches(self, users_votes: Dict[int, List[Vote]]) -> List[int]:
        matched_items = []

        for user_id, votes in users_votes.items():
            for vote in votes:
                if vote.value == 1:
                    self.users_positive_votes[user_id].add(vote)
                    self.all_recommended_items_votes[vote.id] += 1

        for item_id, votes_count in self.all_recommended_items_votes.items():
            if votes_count == len(users_votes):
                matched_items.append(item_id)

        return matched_items


if __name__ == "__main__":
    d_loader = MovieLensDatasetLoader()
    _, ratings_matrix = d_loader.load_data(True)
    ratings_matrix_np = ratings_matrix.to_numpy()

    session_users = [12, 5, 10]
    easer_cached = EaserCached(200)
    easer_cached.fit(ratings_matrix)

    easer_cached.precalculate_scores(session_users)
    recommender = RecommendationEngineIndividualEaser(session_users, ratings_matrix, easer_cached)

    simplePriorityFunction = SimplePriorityFunction(easer_cached)
    redistribution_unit = RedistributionUnit(session_users, simplePriorityFunction)

    window_size = 10
    staticThresholdPolici = ThresholdPolicyStatic(7)
    consensus_mediator = ConsensusMediatorAsyncApproach(session_users, recommender, redistribution_unit, staticThresholdPolici, window_size)

    ##### iterations test

    matching_rounds = 3

    current_round_votes: Dict[int, List[Vote]] = {}
    for i in range(matching_rounds):

        recommendation = consensus_mediator.get_next_round_recommendation(current_round_votes)
        current_round_votes = { user_id: [] for user_id in session_users }

        for user_id in session_users:
            for i, item_id in enumerate(recommendation[user_id]):
                if i < 5:
                    current_round_votes[user_id].append(Vote(item_id, 1))
                else:
                    current_round_votes[user_id].append(Vote(item_id, 0))

