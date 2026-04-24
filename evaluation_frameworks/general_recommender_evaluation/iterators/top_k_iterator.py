
# ===================================
# DESCRIPTION
# ===================================
# Custom iterator for iteration of top K recommended items.
#
# Recommendation algorithm returns this precomputed iterator
# for specific set of items. User then uses this precomputed iterator to retrieve progressively new items.
#
# Limitations: Does not support feedback loop.
#
# NOTE: iterator could be extended for custom ranking/reorder function (a.k.a postprocessing filter).

from typing import List, Optional, Set, Tuple

class TopKIterator:
    def __init__(
        self,
        item_scores: List[Tuple[str, float]],
        exclude: Optional[Set[str]] = None,
    ):
        """
        Args:
            item_scores (List[Tuple[item_id, score]]): List of items with their predicted score.
            exclude (Optional[Set[str]]): Set of item Ids to skip during iteration (e.g. those we already recommended etc...).
        """

        self.exclude = exclude or set()
        self.sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.position < len(self.sorted_items):
            item_id, _ = self.sorted_items[self.position]
            self.position += 1
            if item_id not in self.exclude:
                return item_id

        raise StopIteration