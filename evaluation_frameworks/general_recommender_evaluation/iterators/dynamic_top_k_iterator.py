from typing import List, Set

from evaluation_frameworks.general_recommender_evaluation.iterators.iterator_base import BaseRecommendationIterator


class DynamicTopKIterator(BaseRecommendationIterator):
    def __init__(
        self,
        recommender,
        user_ids: List[int],
        item_pool: List[int],
        exclude: Set[int] = None,
        method: str = "mean",
    ):
        self.recommender = recommender
        self.user_ids = user_ids
        self.item_pool = item_pool
        self.method = method
        self.exclude = exclude or set()
        self.sorted_items = []
        self.position = 0
        self._refresh()

    def _refresh(self):
        scores = self.recommender.score_items(self.user_ids, self.item_pool, self.method)
        self.sorted_items = sorted(
            [(item_id, score) for item_id, score in scores.items() if item_id not in self.exclude],
            key=lambda x: x[1],
            reverse=True,
        )

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.position >= len(self.sorted_items):
            raise StopIteration

        item_id, _ = self.sorted_items[self.position]
        self.position += 1
        return item_id