from typing import List, Tuple
from joblib import Parallel, delayed
from itertools import islice
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score
from surprise import KNNBasic
from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation

# ===================================
# USER-USER CF BASED MODEL
# ===================================

class UserKnnCFModel:

    def __init__(self, trainset, neighbors_k):
        sim_options = {
            'name': 'cosine',
            'user_based': True
        }
        self.algo = KNNBasic(sim_options=sim_options, k=neighbors_k, verbose=False)
        self.algo.fit(trainset)

    def recommend(self, user_id, known_items, k) -> List[Tuple[int, float]]:
        """Recommends top_k items for user_id. Excludes known_items.
        Returns:
            List[Tuple[int, float]]: Returns list of top_k (item_id, score) pairs.
        """

        all_items = set(self.algo.trainset.all_items())
        candidates = all_items - known_items
        predictions = []
        for iid in candidates:
            pred = self.algo.predict(user_id, self.algo.trainset.to_raw_iid(iid))
            predictions.append((iid, pred.est))

        top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
        return top_k


# ===================================
# EVALUATION
# ===================================

class UserKnnCFEvaluation(SurpriseRatingBasedEvaluation):

    def __init__(self, rating_matrix, k, algorithm_k, test_size=0.2, rating_scale=(1, 5)):
        self.k = k  # evaluation top-K
        self.algorithm_k = algorithm_k  # neighbors used in KNN
        super().__init__(rating_matrix, test_size, rating_scale)

    def fit(self):
        model = UserKnnCFModel(self.trainset, self.algorithm_k)
        self.algo = model

    def evaluate(self):
        trainset = self.trainset
        train_matrix = self.train_matrix
        ground_truth = self._build_ground_truth()

        precisions, recalls, ndcgs = [], [], []

        for uid_raw, true_iids in ground_truth.items():

            try:
                uid = trainset.to_inner_uid(uid_raw)
            except ValueError:
                continue

            known_items = set(train_matrix[uid].keys())
            top_k = self.algo.recommend(uid_raw, known_items, self.k)

            predicted_items, predicted_scores = zip(*top_k)

            hits = set(predicted_items) & {trainset.to_inner_iid(iid) for iid in true_iids}

            precisions.append(len(hits) / self.k)
            recalls.append(len(hits) / len(true_iids))

            rel = [1 if i in hits else 0 for i in predicted_items]

            ndcgs.append(ndcg_score([rel], [predicted_scores]))


        return {
            "precision@K": np.mean(precisions) if precisions else 0,
            "recall@K": np.mean(recalls) if precisions else 0,
            "ndcg@K": np.mean(ndcgs) if precisions else 0
        }

    def _build_ground_truth(self):
        """
            Returns a dict mapping each user to a list of test-set items
            they interacted with, excluding items not seen in training.
            (If model wasn't in training, model can't recommend it...,
            this ensures stability in evaluation.)
        """
        gt = defaultdict(list)
        for uid_raw, iid_raw, _ in self.testset:
            if iid_raw in self.trainset._raw2inner_id_items:
                gt[uid_raw].append(iid_raw)
        return gt