import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from restaurant_data.algo_experiments.algos.cb_model import SimpleCBModel
from restaurant_data.algo_experiments.algos.representation_generation import PlacesAPI2RepConvertor

class CBModelEvaluation:
    def __init__(self, rating_matrix, model: SimpleCBModel, k=10, train_ratio=0.8):
        """
        Args:
            rating_matrix (pd.DataFrame): users x items matrix (0 or rating)
            model (SimpleCBModel): model with .data_source.find_similar(...)
            k (int): top-k to evaluate. Default k=10.
            train_ratio (float): proportion of known ratings used for training (rest is test set)
        """
        self.rating_matrix = rating_matrix
        self.model = model
        self.k = k
        self.train_ratio = train_ratio

    def evaluate(self):
        precisions, recalls, ndcgs = [], [], []

        for user_name in self.rating_matrix.index:
            user_row = self.rating_matrix.loc[user_name]
            rated_items = user_row[user_row > 0]

            if len(rated_items) < 2:
                continue  # skip users with not enough data

            train_items = rated_items.sample(frac=self.train_ratio, random_state=42)
            test_items = rated_items.drop(train_items.index)

            if len(test_items) == 0:
                continue

            user_vector = self.create_user_vector_from_train(train_items)

            try:
                recs_df = self.model.data_source.find_similar(user_vector.reshape(1, -1), self.k)
            except Exception as e:
                print(f"User {user_name} skipped: {e}")
                continue

            rec_items = recs_df.index.tolist()
            rec_scores = recs_df["cos_similarities"].tolist()
            ground_truth = set(test_items.index)

            hits = set(rec_items) & ground_truth

            precision = len(hits) / self.k
            recall = len(hits) / len(ground_truth)
            rel = [1 if item in ground_truth else 0 for item in rec_items]
            ndcg = ndcg_score([rel], [rec_scores]) if rel else 0.0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        return {
            f"precision@{self.k}": self.safe_mean(precisions),
            f"recall@{self.k}": self.safe_mean(recalls),
            f"ndcg@{self.k}": self.safe_mean(ndcgs)
        }

    def create_user_vector_from_train(self, train_items: pd.Series) -> np.ndarray:
        representations_dim = len(PlacesAPI2RepConvertor.get_representation_names())
        combined = np.zeros(representations_dim)
        weight_sum = 0

        for place_id, rating in train_items.items():
            if place_id not in self.model.data_source.restaurants_data:
                continue
            place_data = self.model.data_source.restaurants_data[place_id]
            place_vec = np.array(PlacesAPI2RepConvertor.places_api_data_to_vector(place_data))
            combined += rating * place_vec
            weight_sum += rating

        if weight_sum == 0:
            return np.zeros_like(combined)

        user_vector = combined / weight_sum
        norm = np.linalg.norm(user_vector)
        return user_vector / norm if norm > 0 else user_vector

    def safe_mean(self, values):
        return float(np.mean(values)) if len(values) > 0 else 0.0


#class CBModelEvaluation:
#    def __init__(self, rating_matrix, model, k=10):
#        """
#        Args:
#            rating_matrix (pd.DataFrame): users x items matrix (0 or rating)
#            model (SimpleCBModel): model with .recommend(user_id, top_k)
#            k (int): top-k to evaluate. Default k=10.
#        """
#        self.rating_matrix = rating_matrix
#        self.model = model
#        self.k = k
#
#    def evaluate(self):
#        precisions, recalls, ndcgs = [], [], []
#
#        for user_name in self.rating_matrix.index:
#            user_row = self.rating_matrix.loc[user_name]
#            ground_truth = set(user_row[user_row > 0].index)
#
#            if not ground_truth:
#                continue
#
#            try:
#                recs = self.model.recommend_by_user_name(user_name, self.k)
#            except Exception as e:
#                print(f"User {user_name} skipped: {e}")
#                continue
#
#            rec_items, rec_scores = zip(*recs)
#
#            # find number of positive recommendations
#            hits = set(rec_items) & ground_truth
#
#            # the actual count in top-k
#            precision = len(hits) / self.k
#            # the actual count in all
#            recall = len(hits) / len(ground_truth)
#
#            # sorting by the similarity -> find the relevance
#            rel = [1 if item in ground_truth else 0 for item in rec_items]
#            ndcg = ndcg_score([rel], [rec_scores]) if rel else 0.0
#
#            precisions.append(precision)
#            recalls.append(recall)
#            ndcgs.append(ndcg)
#
#        return {
#            f"precision@{self.k}": self.safe_mean(precisions),
#            f"recall@{self.k}": self.safe_mean(recalls),
#            f"ndcg@{self.k}": self.safe_mean(ndcgs)
#        }
#
#    def safe_mean(self, values):
#        return float(np.mean(values)) if len(values) > 0 else 0.0