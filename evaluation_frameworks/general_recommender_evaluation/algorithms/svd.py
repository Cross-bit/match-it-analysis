from typing import Dict
import pandas as pd
from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoBase
from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation
from sklearn.metrics import ndcg_score
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from scipy.sparse import csr_matrix
import numpy as np

# ===================================
# DESCRIPTION
# ===================================
# Implementation of SVD evaluation using surprise library.
#

class SurpriseSVDRecommender(RecAlgoBase):

    def __init__(self, embeddings_dims=50, random_state=42):
        self._embeddings_dims = embeddings_dims
        self._random_state = random_state
        pass


    def _csr_to_dataframe(self, csr, user_id_map:Dict[int, int], item_id_map:Dict[int, int]):
        """
        Converts a CSR matrix into a DataFrame with columns: userId, itemId, rating.
        Optionally takes external ID maps.

        Args:
            csr: csr_matrix, shape [n_users, n_items]
            user_id_map: dict {row_index: user_id} – optional
            item_id_map: dict {col_index: item_id} – optional

        Returns:
            pd.DataFrame with columns: userId, itemId, rating
        """
        coo = csr.tocoo()
        df = pd.DataFrame({
            "userId": coo.row,
            "itemId": coo.col,
            "rating": coo.data
        })

        if user_id_map:
            df["userId"] = df["userId"].map(user_id_map)
        if item_id_map:
            df["itemId"] = df["itemId"].map(item_id_map)

        return df

    def fit(self, ratings_matrix_csr: csr_matrix, user_id_map: Dict[int, int], item_id_map: Dict[int, int]) -> "RecAlgoBase":

        print("Building ratings dataframe...")
        ratings_long_format: pd.DataFrame = self._csr_to_dataframe(ratings_matrix_csr, user_id_map, item_id_map)

        print("Converting to Surprise dataset...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_long_format[["userId", "itemId", "rating"]], reader)
        self._current_trainset = data.build_full_trainset()

        print("Training SVD...")
        self._model = SVD(n_factors=self._embeddings_dims, random_state=self._random_state)
        self._model = self._model.fit(self._current_trainset)

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return self._model.predict(user_id, item_id).est






class SVDPrecisionEvaluation(SurpriseRatingBasedEvaluation):
    def __init__(self, rating_matrix, k, test_size=0.2, rating_scale=(1, 5), n_factors=50):
        self.k = k
        self.n_factors = n_factors
        super().__init__(rating_matrix, test_size, rating_scale)

    def fit(self):
        algo = SVD(n_factors=self.n_factors, random_state=42)
        algo.fit(self.trainset)
        self.algo = algo

    def evaluate(self):
        trainset = self.trainset
        ground_truth = self._build_ground_truth()

        precisions, recalls, ndcgs = [], [], []

        for uid_raw, true_iids in ground_truth.items():
            try:
                uid_inner = trainset.to_inner_uid(uid_raw)
            except ValueError:
                continue

            # Score all items for this user.
            all_items = set(trainset.all_items())
            known_items = set(j for (j, _) in self.train_matrix[uid_inner].items())
            candidate_items = list(all_items - known_items)

            predictions = []
            for iid in candidate_items:
                iid_raw = trainset.to_raw_iid(iid)
                pred = self.algo.predict(uid_raw, iid_raw)
                predictions.append((iid, pred.est))

            # order by score
            top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:self.k]
            top_k_iids, top_k_scores = zip(*top_k) if top_k else ([], [])

            # metric calculations
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