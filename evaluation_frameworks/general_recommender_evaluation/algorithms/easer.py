#!/bin/python3
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional,Set
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.metrics import ndcg_score
from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoBase, RecAlgoFull
from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation
from evaluation_frameworks.general_recommender_evaluation.iterators.top_k_iterator import TopKIterator

# ===================================
# DESCRIPTION
# ===================================
# Implementation of Easer algorithm according to the original paper
# https://arxiv.org/pdf/1905.03375
#
# + evaluations
#

from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation

# ===================================
# DESCRIPTION
# ===================================
# Implementation of Easer algorithm according to the original paper
# https://arxiv.org/pdf/1905.03375
#
# + evaluations
#

class EaserSparse(RecAlgoFull):
    def __init__(self, l2: float = 0.5):
        self.l2 = l2
        self.B = None  # item-item similarity matrix

        self._user_id_to_internal_row_index: Dict[int, int] = {}
        self._item_id_to_internal_col_index: Dict[int, int] = {}
        self._internal_col_index_to_item_id: Dict[int, int] = {}
        self._ratings_matrix: Optional[csr_matrix] = None
        self._cached_scores: Dict[int, np.ndarray] = {}

    def fit(self, ratings: csr_matrix, user_id_map: Dict[int, int], item_id_map: Dict[int, int]) -> "RecAlgoFull":
        """
        Fit the EASE model using a sparse user-item matrix.

        Args:
            ratings (csr_matrix): User-item matrix.
            user_id_map (Dict[int, int]): CSR row index → external user_id
            item_id_map (Dict[int, int]): CSR col index → external item_id
        """

        self._user_id_to_internal_row_index = {v: k for k, v in user_id_map.items()}
        self._item_id_to_internal_col_index = {v: k for k, v in item_id_map.items()}

        self._internal_col_index_to_item_id = item_id_map

        self._ratings_matrix = ratings

        G = ratings.T @ ratings  # item-item co-occurrence
        G = G.toarray()
        diag = np.arange(G.shape[0])
        G[diag, diag] += self.l2

        P = np.linalg.inv(G)
        B = -P / np.diag(P)[:, None]
        B[diag, diag] = 0

        self.B = B

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if self._ratings_matrix is None:
            raise Exception("No model data found. Call fit() first.")

        # Map external IDs to internal row/column indices.
        if user_id not in self._user_id_to_internal_row_index:
            raise ValueError(f"Unknown user_id: {user_id}")
        if item_id not in self._item_id_to_internal_col_index:
            raise ValueError(f"Unknown item_id: {item_id}")

        row_idx  = self._user_id_to_internal_row_index[user_id]
        col_idx  = self._item_id_to_internal_col_index[item_id]

        user_vector = self._ratings_matrix[row_idx, :]
        return float(user_vector @ self.B[:, col_idx])

    def precalculate_scores(self, user_ids: List[int]):
        """
        Compute and cache predicted scores for users.
        """
        for user_id in user_ids:
            if user_id not in self._user_id_to_internal_row_index:
                continue

            if user_id not in self._cached_scores:
                row_idx = self._user_id_to_internal_row_index[user_id]
                user_vector = self._ratings_matrix[row_idx, :].toarray().flatten()
                self._cached_scores[user_id] = user_vector @ self.B

    def clear_cached_scores(self, user_id: Optional[int] = -1):
        if user_id == -1:
            self._cached_scores: Dict[int, np.ndarray] = {} # reset entire cache
        elif user_id in self._cached_scores:
            del self._cached_scores[user_id]
        else:
            warnings.warn("Clearing cache: User id {user_id} was not in cache, skipping.")

    def get_cached_prediction(self, user_id: int, item_id: int) -> float:
        """
        Get the prediction score for a user-item pair.

        Raises:
            Exception if user or item not found in model.
        """
        if user_id not in self._cached_scores:
            raise Exception(f"User {user_id} not cached.")
        if item_id not in self._item_id_to_internal_col_index:
            raise Exception(f"Item {item_id} not known.")

        item_idx = self._item_id_to_internal_col_index[item_id]
        return self._cached_scores[user_id][item_idx]

    def get_item_scores(self, user_vector: np.ndarray) -> np.ndarray:
        """
        Returns predicted item scores for the given user vector.
        Args:
            user_vector (np.ndarray): Dense user vector (shape: [n_items])
        Returns:
            np.ndarray: Dense score vector (shape: [n_items])
        """
        if self.B is None:
            raise Exception("Model not fitted yet.")

        return user_vector @ self.B

    def get_user_vector(self, user_id: int) -> np.ndarray:
        """
        Returns a dense user vector from the internal CSR matrix.
        Args:
            user_id (int): External user ID.
        Returns:
            np.ndarray: Dense user vector (shape: [n_items])
        """

        if self._ratings_matrix is None:
            raise Exception("Model not fitted yet.")
        if user_id not in self._user_id_to_internal_row_index:
            raise ValueError(f"User {user_id} not known.")

        row_idx = self._user_id_to_internal_row_index[user_id]
        return self._ratings_matrix[row_idx, :].toarray().flatten()

    def item_id_to_index(self, item_id: int) -> int:
        """
        Maps external item_id to internal column index.

        Args:
            item_id (int): External item ID

        Returns:
            int: Internal index in the score vector / matrix
        """
        if item_id not in self._item_id_to_internal_col_index:
            raise ValueError(f"Unknown item_id: {item_id}")
        return self._item_id_to_internal_col_index[item_id]

    def index_to_item_id(self, index: int) -> int:
        """
        Maps internal column index back to external item_id.

        Args:
            index (int): Internal item index

        Returns:
            int: External item ID
        """
        if index not in self._internal_col_index_to_item_id:
            raise ValueError(f"Unknown internal item index: {index}")
        return self._internal_col_index_to_item_id[index]


    def top_k_iterator(self, user_id: int, exclude: Optional[Set[int]] = None) -> TopKIterator:
        """
        Iterator over top-K items by predicted score for a given user.

        Args:
            user_id (int): ID of the user.
            exclude (set): Optional set of item IDs to exclude.

        Returns:
            TopKIterator: sorted descending (item_id, score) pairs.
        """
        if user_id not in self._cached_scores:
            row_idx = self._user_id_to_internal_row_index[user_id]
            user_vector = self._ratings_matrix[row_idx, :].toarray().flatten()
            scores = self.get_item_scores(user_vector)
        else:
            scores = self._cached_scores[user_id]

        item_scores = [(self._internal_col_index_to_item_id[i], scores[i]) for i in range(len(scores))]
        return TopKIterator(item_scores, exclude=exclude)

class EaserBase:
    def __init__(self, l2=0.5):
        """
        Base class for the EASE algorithm.
        Args:
            l2 (float): L2 regularization parameter.
        """
        self.l2 = l2
        self.B = None  # item-item similarity matrix

        self._user_to_row = {}
        self._item_to_col = {}
        self._col_to_item = {}
        self._ratings_matrix = None

    def fit(self, ratings_df: pd.DataFrame) -> np.ndarray:
        """
        Fits the EASE model to the user-item rating DataFrame.

        Args:
            ratings_df (pd.DataFrame): Rows = users, Columns = items

        Returns:
            np.ndarray: Learned item-item similarity matrix B.
        """
        self._user_to_row = {uid: idx for idx, uid in enumerate(ratings_df.index)}
        self._item_to_col = {iid: idx for idx, iid in enumerate(ratings_df.columns)}
        self._col_to_item = {idx: iid for iid, idx in self._item_to_col.items()}

        self._ratings_matrix = ratings_df.to_numpy(dtype=np.float64)

        G = self.find_coocurrence_matrix(self._ratings_matrix)
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.l2  # L2 regularization

        P = np.linalg.inv(G)
        self.B = -P / np.diag(P)[:, None]
        self.B[diag_indices] = 0

        return self.B

    def find_coocurrence_matrix(self, ratings_matrix: np.ndarray) -> np.ndarray:
        """
        Computes item-item co-occurrence matrix G.

        Args:
            ratings_matrix (np.ndarray): NumPy user-item matrix.

        Returns:
            np.ndarray: Co-occurrence matrix G.
        """
        X_sparse = csr_matrix(ratings_matrix)
        G_sparse = X_sparse.T @ X_sparse
        return G_sparse.toarray()

    def predict(self, user_id, item_id) -> float:
        """
        Predicts the score for given user_id and item_id.

        Args:
            user_id: ID of the user (must exist in training data).
            item_id: ID of the item (must exist in training data).

        Returns:
            float: Predicted score.
        """
        if self.B is None:
            raise Exception("Model not fitted!")

        try:
            row_idx = self._user_to_row[user_id]
            col_idx = self._item_to_col[item_id]
        except KeyError as e:
            raise ValueError(f"Unknown user or item: {e}")

        user_vector = self._ratings_matrix[row_idx, :]
        return np.dot(user_vector, self.B[:, col_idx])


# ===========================================
# EVALUATIONs
# ===========================================

class EaserEvaluation(SurpriseRatingBasedEvaluation):

    def __init__(self, rating_matrix, k, test_size = 0.2, rating_scale=(1, 5), regularization = 200):
        """_summary_

        Args:
            rating_matrix (_type_): _description_
            k (_type_): Evaluate
            test_size (float, optional): _description_. Defaults to 0.2.
            rating_scale (tuple, optional): _description_. Defaults to (1, 5).
            regularization (int, optional): _description_. Defaults to 100.
        """
        self.regularization = regularization
        self.k = k
        super().__init__(rating_matrix, test_size, rating_scale)

    def fit(self):
        ratings_df = pd.DataFrame(self.train_matrix)

        easer = EaserBase(self.regularization)
        easer.fit(ratings_df)
        self.algo = easer

    def evaluate(self):
        B = self.algo.B

        train_matrix = self.train_matrix
        trainset = self.trainset

        ground_truth = self._build_ground_truth() # All the items users interacted with in the test set

        precisions, recalls, ndcgs = [], [], []

        for uid_raw, true_iids in ground_truth.items():
            try:
                uid = trainset.to_inner_uid(uid_raw)
            except ValueError:
                continue  # user not in trainset


            x_u = np.zeros(train_matrix.shape[1])
            for (_, iid), rating in train_matrix[uid].items(): # train_matrix[uid] returns dict -> conversion needed
                x_u[iid] = rating

            # find all items scores
            scores = x_u @ B

            # suppress items from the train set
            known_items = set(train_matrix[uid].keys())
            scores[list(known_items)] = -np.inf  # mask known items


            true_inner_iids = {
                trainset.to_inner_iid(iid)
                for iid in true_iids
                if iid in trainset._raw2inner_id_items
            }

            # select only top k
            top_k = np.argsort(scores)[-self.k:][::-1]
            predicted_scores = [scores[i] for i in top_k]

            hits = set(top_k) & true_inner_iids

            # Metrics
            precisions.append(len(hits) / self.k)
            recalls.append(len(hits) / len(true_iids))
            rel = [1 if i in true_inner_iids else 0 for i in top_k]
            ndcgs.append(ndcg_score([rel], [predicted_scores]))

        return {
            "precision@K": np.mean(precisions),
            "recall@K": np.mean(recalls),
            "ndcg@K": np.mean(ndcgs)
        }

    def _build_ground_truth(self):
        gt = defaultdict(list)
        for uid_raw, iid_raw, _ in self.testset:
            gt[uid_raw].append(iid_raw)
        return gt

