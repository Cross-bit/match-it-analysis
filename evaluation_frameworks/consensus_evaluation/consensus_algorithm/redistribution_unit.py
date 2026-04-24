"""
`redistribution_unit` — per-user priority queues and policies for async redistribution.

When the async mediator recommends a mix of **new** items and **re-offered**
items from earlier rounds, this module decides **which** held-over items enter
each user's queue and **in what order** they should surface.

Contents:

- ``RedistributionContext`` — minimal read-only interface exposed to priority
  functions (vote totals, round index, queue sizes).
- ``PriorityFunction`` / ``SimplePriorityFunction`` / ``MultiplicativePriorityNormalized``
  — pluggable scoring: combine cached recommender scores with group vote
  signals to produce a float priority per ``(user_id, item_id)``.
- ``RedistributionUnit`` — concrete context + state:

  - one ``SimplePriorityQueue`` per user (pending redistribution items),
  - bookkeeping of all votes and who liked which item (for vote totals),
  - ``update_voted_items`` to ingest a round of ``Vote`` objects: drop stale
    items, enqueue newly liked items the user has not seen yet,
  - ``get_redistributed_items`` to pop up to ``t`` items for the next slate.

Depends on ``RecAlgoCached`` (typically ``EaserCached``) for batched score
lookups and on ``Vote`` from ``models.py``. Wired from ``consensus_mediator``
for async policies that use redistribution + threshold splitting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set

import numpy as np

from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoCached
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer_cached import EaserCached
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.models import Vote
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.priority_queue import SimplePriorityQueue


class RedistributionContext(ABC):
    @abstractmethod
    def get_item_total_votes(self, item_id: int):
        """ How many users voted over the given item."""
        pass

    @abstractmethod
    def get_current_round(self):
        """Returns current voting round."""
        pass

    @abstractmethod
    def get_user_queue_size(self, user_id: int):
        """Returns current voting round."""
        pass

class PriorityFunction(ABC):
    @abstractmethod
    def get_priority(self, user_id: int, item_id: int, context: RedistributionContext) -> float:
        pass

    def get_metadata(self):
        return {
            "type": self.__class__.__name__,
            "general_desc": "Priority function of redistribution unit."
        }


class SimplePriorityFunction(PriorityFunction):

    def __init__(self, group: List[int], algorithm: RecAlgoCached):
        self.algo: RecAlgoCached = algorithm
        self.algo.precalculate_scores(group)
        self.group = group

    def get_priority(self, user_id: int, item_id: int, context: RedistributionContext) -> float:
        user_rating = self.algo.get_cached_prediction(user_id, item_id)
        priority = user_rating * context.get_item_total_votes(item_id)

        return priority

    def get_metadata(self):
        base = super().get_metadata()
        base["algo"] = self.algo.__class__.__name__
        return base


class MultiplicativePriorityNormalized(PriorityFunction):
    """``rating * votes`` chování s měřítky: rating v [0,1] (min–max přes celý vektor uživatele), votes / |G|."""

    _EPS = 1e-12

    def __init__(self, group: List[int], algorithm: RecAlgoCached):
        self.algo: RecAlgoCached = algorithm
        self.group = group
        self._g = max(1, len(group))
        self.algo.precalculate_scores(group)
        self._r_min: Dict[int, float] = {}
        self._r_max: Dict[int, float] = {}
        cached = getattr(self.algo, "_cached_scores", None)
        if not isinstance(cached, dict):
            raise TypeError(
                "MultiplicativePriorityNormalized potřebuje algoritmus s atributem _cached_scores (např. EaserCached)."
            )
        for uid in group:
            vec = cached.get(uid)
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float64)
            self._r_min[uid] = float(np.min(arr))
            self._r_max[uid] = float(np.max(arr))

    def get_priority(self, user_id: int, item_id: int, context: RedistributionContext) -> float:
        r = self.algo.get_cached_prediction(user_id, item_id)
        lo = self._r_min.get(user_id)
        hi = self._r_max.get(user_id)
        if lo is None or hi is None or hi - lo < self._EPS:
            r_norm = 0.5
        else:
            r_norm = float(np.clip((r - lo) / (hi - lo), 0.0, 1.0))

        votes = context.get_item_total_votes(item_id)
        v_norm = min(1.0, float(votes) / float(self._g))

        return r_norm * v_norm

    def get_metadata(self):
        base = super().get_metadata()
        base["algo"] = self.algo.__class__.__name__
        base["rating_scale"] = "per_user_min_max_full_vector"
        base["vote_scale"] = f"divide_by_group_size_{self._g}"
        return base


class STSGroupIndividualPriority(PriorityFunction):
    """
    Priority for redistribution that uses:
    - dynamic individual score from STS-like engine
    - multiplied by social signal (positive votes over item)
    """

    def __init__(self, engine_with_individual_scores):
        self.engine = engine_with_individual_scores

    def get_priority(self, user_id: int, item_id: int, context: RedistributionContext) -> float:
        ind_score = float(self.engine.get_individual_item_score(user_id, item_id))
        votes = float(context.get_item_total_votes(item_id))
        return ind_score * votes

    def get_metadata(self):
        base = super().get_metadata()
        base["engine"] = getattr(self.engine, "__class__", type(self.engine)).__name__
        base["formula"] = "individual_dynamic_score * positive_votes"
        return base


#class SimplePriorityFunctionWithRandom(SimplePriorityFunction):
#    def get_priority(self, user_id: int, item_id: int, context: RedistributionContext) -> float:
#        pass


class RedistributionUnit(RedistributionContext):

    def __init__(self, users_ids: List[int], priority_function: PriorityFunction):
        self.items_Queue = { user_id: SimplePriorityQueue() for user_id in users_ids }
        self.user_all_voted_items = { user_id: set() for user_id in users_ids } # mapping user id => all items user voted for (liked, disliked, neutral, .. => all responses)
        self.liked_items_by_user_map = {} # maps item id => users that liked this item so far
        self.priority_function = priority_function
        self.round_counter = 0


    def get_redistributed_items_all(self, users_ids: List[int]) -> Dict[int, List[int]]:
        res = { user_id: [] for user_id in users_ids }
        for user_id in users_ids:
            res[user_id] = self.get_redistributed_items(user_id)

        return res

    def get_user_redistribution_queue_size(self, user_id: int) -> bool:
        return len(self.items_Queue[user_id])

    def get_redistributed_items(self, user_id: int, t: int) -> List[int]:
        """Tries to get t redistributed items for a user.
            If not enough items are available returns all possible.
        Args:
            user_id (str): User ID.
            t (int): The t parameter from the async algorithm definition.
        """

        # Pop t items from the user queue
        recommendation_res = []
        for _ in range(t):
            item_id = self.items_Queue[user_id].pop()
            recommendation_res.append(item_id)

        return recommendation_res

    def get_user_queue_size(self, user_id: int):
        """Returns current voting round."""
        return len(self.items_Queue[user_id])


    def update_voted_items(self, users_votes_all: Dict[int, List[Vote]]) -> None:
        """Redistributes user items from the previous rounds to the current users."""

        if users_votes_all == {}:
            return

        # suppose only positive feedback items here
        users_positive_votes_only = self._filter_positive_votes_only(users_votes_all)

        # aggregate all the items users voted positively in current round
        all_liked_items_ids_in_current_round = {
            item.id for items in users_positive_votes_only.values()
            for item in items
        }

        # all items ids voted over in current round
        all_voted_items_ids_in_current_round = {
            item.id for items in users_votes_all.values()
            for item in items
        }

        for user_id, positive_votes_in_cur_round in users_positive_votes_only.items():
            # update cached votes
            self._update_positively_voted_items(user_id, positive_votes_in_cur_round) # (only items user liked)
            self._update_user_voted_items(user_id, users_votes_all[user_id]) # (all items user voted so far!!)

        for user_id, positive_votes_in_cur_round in users_positive_votes_only.items():

            # 1. Remove outdated items

            # remove all items that are not valid for the user anymore (items that others rejected/gave neutral, while current user did not see them so far)
            user_items_to_discard = (all_voted_items_ids_in_current_round - all_liked_items_ids_in_current_round)
            self.items_Queue[user_id].discard_many(user_items_to_discard)

            # 2. Redistribute all items user did not see

            # get all items user did not vote over so far
            user_all_voted_items_ids_so_far = { item.id for item in self.user_all_voted_items[user_id] }
            items_to_redistribute_to_user = all_liked_items_ids_in_current_round - user_all_voted_items_ids_so_far

            # update user priority queue
            self._enqueue_user_items(user_id, items_to_redistribute_to_user)


        self.round_counter += 1

    def _filter_positive_votes_only(self, users_votes_all: Dict[str, List[Vote]]) -> Dict[int, List[Vote]]:
        """Filter only items over which users voted positively (there is a chance for a match)"""
        return {
            user_id: [item for item in items if item.value == 1]
            for user_id, items in users_votes_all.items()
        }

    def _enqueue_user_items(self, user_id: int, items_to_redistribute: Set[int]):

        for item_id in items_to_redistribute:
            # find priority
            item_priority = self._find_item_priority(user_id, item_id)
            self.items_Queue[user_id].add_or_update(item_id, item_priority)

    def _find_item_priority(self, user_id: int, item_id: int) -> float:
        self.priority_function.get_priority(user_id, item_id, self)
        item_priority = self.priority_function.get_priority(user_id, item_id, self)
        return item_priority

    def _update_positively_voted_items(self, user_id: int, positively_voted_items: List[Vote]):
        for item in positively_voted_items:
            if item.id not in self.liked_items_by_user_map:
                self.liked_items_by_user_map[item.id] = set()
            self.liked_items_by_user_map[item.id].add(user_id)

    def _update_user_voted_items(self, user_id: int, voted_items: List[Vote]):
        self.user_all_voted_items[user_id].update(voted_items)

    def get_item_total_votes(self, item_id: int):
        return len(self.liked_items_by_user_map[item_id])

    def get_current_round(self):
        return self.round_counter