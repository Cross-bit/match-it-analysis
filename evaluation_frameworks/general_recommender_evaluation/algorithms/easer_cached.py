#!/bin/python3
from typing import List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from scipy.sparse import csr_matrix
from collections import defaultdict

from typing import Dict, List, Optional, Set, Tuple
from scipy.sparse import csr_matrix
import numpy as np

from evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserBase
from evaluation_frameworks.general_recommender_evaluation.iterators.top_k_iterator import TopKIterator

# =================================
# DESCRIPTION
# =================================
# Efficient implementation of easer that stores precomputed values
# https://arxiv.org/pdf/1905.03375
#
# + evaluations
#

class EaserCached(EaserBase):
    def __init__(self, l2=0.5):
        super().__init__(l2)
        self._cached_scores = {}  # user_id -> prediction vector

    def precalculate_scores(self, user_ids: List[int]):
        """
        Precomputes and caches predicted scores for given users.

        Args:
            user_ids (list): List of user IDs to precompute for.
        """
        if self.B is None:
            raise Exception("Model not fitted. Call fit() first.")

        for user_id in user_ids:
            if user_id not in self._user_to_row:
                continue  # skip unknown user

            row_idx = self._user_to_row[user_id]
            user_vector = self._ratings_matrix[row_idx, :]
            scores = np.dot(user_vector, self.B)
            self._cached_scores[user_id] = scores

    def get_cached_prediction(self, user_id: int, item_id: int) -> float:
        """
        Returns the precomputed prediction score for given user and item.
        """
        if user_id not in self._cached_scores:
            raise Exception(f"User {user_id} not cached. Call precalculate_scores() first.")

        if item_id not in self._item_to_col:
            raise Exception(f"Item {item_id} unknown.")

        item_idx = self._item_to_col[item_id]
        return self._cached_scores[user_id][item_idx]

    def top_k_iterator(self, user_id: int, exclude: Optional[Set[int]] = None) -> TopKIterator:
        """
        Returns an iterator over items sorted by score (high to low).
        """
        if user_id not in self._cached_scores:
            raise Exception(f"User {user_id} not cached. Call precalculate_scores() first.")

        scores = self._cached_scores[user_id]

        item_scores = [
            (int(self._col_to_item[i]), score)
            for i, score in enumerate(scores)
        ]

        return TopKIterator(item_scores, exclude=exclude)