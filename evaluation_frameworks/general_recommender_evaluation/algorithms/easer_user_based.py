#!/bin/python3
import warnings
import pandas as pd
import numpy as np


from sklearn.metrics import ndcg_score
from scipy.sparse import csr_matrix
from collections import defaultdict


from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation

# ===================================
# DESCRIPTION
# ===================================
# Implementation of edited Easer algorithm with user based G \in R^|U|x|U|.
#
# Original paper idea:
# Matrix G is basically "popularity" matrix of how many users rated items i,j  together.
# Filtering the noise from other items using Inverse gives pure i,j relationships in precision matrix P.
# (If we imagine matrix as complete graph, we obtain cleaned graph from the wrong transitivity
#  we don't want.)
#
# User based idea:
#  Matrix G is basically a count matrix, of how many items users i, j rated together.
#  (Without inverse computing B yields asymmetric Jaccard similarity of user i to user j)
#  Finding inverse leaves us with pure relation ships in P (without noise).
#  This naturally finds a "meta-groups" in the original dataset.
#  (Which can be visualised as an social graph of relationships,
#   thus may discover hidden relationships among people)
#
#

class EaserBaseRaw:
    def __init__(self, l2 = 0.5):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2
        )

        """ Base class for the ease algorithm.

        Args:
            l2 (float, optional): L2 regularisation.
            Should be chosen to support numerical stability of the inverse
            operation of the matrix G.
        """
        self.l2 = l2
        self.B = None # item-item similarity matrix

    def fit(self, ratings_matrix_X: np.ndarray) -> np.ndarray:
        """ Finds matrix B using closed formula from the original paper.
        Args:
            X: User ratings matrix.
        Returns:
            np.ndarray: Returns model matrix B
        """
        G = self.find_coocurance_matrix(ratings_matrix_X)

        diag_indices = np.diag_indices(G.shape[0])

        G[diag_indices] += self.l2  # L2 regularization

        P = np.linalg.inv(G)

        self.B = -P / np.diag(P)

        self.B[diag_indices] = 0  # zero diagonal

        return self.B

    def find_coocurance_matrix(self, ratings_matrix_X: np.ndarray) -> np.ndarray:
        """Computes coocurance matrix for the given ratings_matrix_X.

        Args:
            ratings_matrix_X (np.ndarray): Ratings matrix X, user_id x item_id.

        Returns:
            np.ndarray: Grams coocurance matrix G (from the paper).
        """

        X_train_sparse = csr_matrix(ratings_matrix_X.astype(np.float64))
        G_sparse: csr_matrix = X_train_sparse.T @ X_train_sparse

        return G_sparse.toarray()

    def predict(self, ratings_matrix_X: np.ndarray, user_id: int, item_id: int) -> np.array:
        """ Predicts ratings for the user, item pair.

        Args:
            ratings_matrix_X np.ndarray: Standard ratings matrix UxI.
            user_id int: User id.
            item_id int: Item id.

        Raises:
            Exception: If B not exists raises exception. (The fitting must be performed first.)

        Returns:
            int: Returns predicted score for user_id and item_id.
        """

        if (self.B is None):
            raise Exception("No data fitted!")

        return np.dot(ratings_matrix_X[user_id, :], self.B[:, item_id])

class EaserUserBased(EaserBaseRaw):

    def __init__(self, l2 = 0.5):
        super().__init__(l2)

    def find_coocurance_matrix(self, ratings_matrix_X: np.ndarray) -> np.ndarray:
        """Computes coocurance matrix for the given ratings_matrix_X.

        Args:
            ratings_matrix_X (np.ndarray): Ratings matrix X, user_id x item_id.

        Returns:
            np.ndarray: Grams coocurance matrix G (from the paper).
        """

        X_train_sparse = csr_matrix(ratings_matrix_X.astype(np.float64))
        G_sparse: csr_matrix = X_train_sparse @ X_train_sparse.T

        return G_sparse.toarray()


    def predict(self, ratings_matrix_X, user_id, item_id):
        if (self.B == None):
            raise Exception("No data fitted!")

        return np.dot(ratings_matrix_X.T[item_id, :], self.B[:, user_id])


# ===========================================
# EVALUATIONs
# ===========================================

class EaserUserBasedPrecisionEvaluation(SurpriseRatingBasedEvaluation):

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
        easer = EaserUserBased(self.regularization)
        G_matrix = easer.find_coocurance_matrix(self.train_matrix)

        pd.DataFrame(data=easer.fit(G_matrix))

        self.algo = easer

    def evaluate(self):
        B = self.algo.B  # now user-user similarity matrix
        train_matrix = self.train_matrix
        trainset = self.trainset

        ground_truth = self._build_ground_truth()
        precisions, recalls, ndcgs = [], [], []

        for uid_raw, true_iids in ground_truth.items():
            try:
                uid = trainset.to_inner_uid(uid_raw)
            except ValueError:
                continue

            # Predict scores using user-user similarity
            scores = B[uid] @ train_matrix.toarray()

            # suppress known items from training
            known_items = set(train_matrix[uid].keys())
            scores[list(known_items)] = -np.inf

            top_k = np.argsort(scores)[-self.k:][::-1]
            true_inner_iids = {trainset.to_inner_iid(iid) for iid in true_iids}
            hits = set(top_k) & true_inner_iids

            precisions.append(len(hits) / self.k)
            recalls.append(len(hits) / len(true_iids))
            rel = [1 if i in true_inner_iids else 0 for i in top_k]
            ndcgs.append(ndcg_score([rel], [rel]))

        return {
            "precision@K": np.mean(precisions),
            "recall@K": np.mean(recalls),
            "ndcg@K": np.mean(ndcgs)
        }

    def _build_ground_truth(self):
        """Returns dict user->[list of items user interacted with in testset]"""
        gt = defaultdict(list)
        for uid_raw, iid_raw, _ in self.testset:
            if iid_raw in self.trainset._raw2inner_id_items:
                gt[uid_raw].append(iid_raw)
        return gt
