from itertools import islice
from joblib import Parallel, delayed
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score
from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation

# ===================================
# DESCRIPTION
# ===================================
# Implementation of trivial popularity baseline prediction algorithm.
#
#

class PopularityBaseline:

    def fit(self, train_matrix):
        item_counts = np.array(train_matrix.sum(axis=0)).ravel()
        self.item_scores = item_counts
        self.top_items = np.argsort(item_counts)[::-1]

    def recommend(self, user_id, known_items, k):
        recs = [(i, self.item_scores[i]) for i in self.top_items if i not in known_items]
        return recs[:k]

# ===========================================
# EVALUATIONs
# ===========================================

class PopularityEvaluation(SurpriseRatingBasedEvaluation):

    def __init__(self, rating_matrix, k, test_size=0.2, rating_scale=(1, 5)):
        self.k = k
        super().__init__(rating_matrix, test_size, rating_scale)

    def fit(self):
        model = PopularityBaseline()
        model.fit(self.train_matrix)
        self.algo = model

    def evaluate(self):
        train_matrix = self.train_matrix
        trainset = self.trainset
        ground_truth = self._build_ground_truth()

        precisions, recalls, ndcgs = [], [], []

        for uid_raw, true_iids in ground_truth.items():
            try:
                uid = trainset.to_inner_uid(uid_raw)
            except ValueError:
                continue

            known_items = set(train_matrix[uid].keys())
            top_k = self.algo.recommend(uid, known_items, self.k)
            top_k_iids, top_k_scores = zip(*top_k) if top_k else ([], [])

            hits = set(top_k_iids) & {trainset.to_inner_iid(iid) for iid in true_iids}

            precisions.append(len(hits) / self.k)
            recalls.append(len(hits) / len(true_iids))
            rel = [1 if i in hits else 0 for i in top_k_iids]
            ndcgs.append(ndcg_score([rel], [top_k_scores]))

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
