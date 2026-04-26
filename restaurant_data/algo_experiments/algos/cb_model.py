#!/bin/python3
from abc import ABC
import os
import sys
import numpy as np
import pandas as pd

from restaurant_data.algo_experiments.algos.representation_generation import PlacesAPI2RepConvertor
from latex_utils.latex_table_generator import LaTeXTableGenerator, LaTeXTableGeneratorSIUnitx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics.pairwise import cosine_similarity
from dataset.data_access import DatasetLoader


class CB_DataSource(ABC):
    """
    Representation of some CB data datasource.
    """

    def get_all_user_representation(self):
        pass

    def find_similar(self, user_vector: np.array):
        # this may contain query to vector database to efficiently fetch all similar vectors
        pass

class Local_CB_DataSource(CB_DataSource):

    def __init__(self, min_number_of_users_in_dataset = 1):
        self.data_loader = DatasetLoader()
        full_rating_matrix = self.data_loader.load_ratings_matrix()
        self.ratings_matrix: pd.DataFrame = self.data_loader.remove_users_from_ratings_matrix_by_ratings_count(full_rating_matrix, min_number_of_users_in_dataset, True)

        removed_places_ids = set(full_rating_matrix.columns.to_list()) - set(self.ratings_matrix.columns.tolist())
        self.restaurants_data: Dict = self._load_data_by_id(removed_places_ids)

        self.restaurants: pd.DataFrame = self._create_places_representations() # "the database or restaurants"
        self.users = self._create_user_representations() # "the database of users"
        super().__init__()

    def _load_data_by_id(self, ids_to_filter = []):
        data_raw = self.data_loader.load_data_raw()["places"]
        result = {}
        for rest_data in data_raw:
            rest_id = rest_data['id']
            if (rest_id in ids_to_filter):
                continue

            result[rest_id] = rest_data
        return result

    def _create_places_representations(self):
        data = []
        for id, restaurant in self.restaurants_data.items():
            representation = np.array(PlacesAPI2RepConvertor.places_api_data_to_vector(restaurant))
            representation_normed = self._safe_norm(representation)
            record = [id] + representation_normed.tolist()
            data.append(record)

        data_header = ["id"] + PlacesAPI2RepConvertor.get_representation_names()
        places_representations = pd.DataFrame(data, columns=data_header)
        places_representations.fillna(0, inplace=True)
        return pd.DataFrame(places_representations, columns=data_header)

    def _create_user_representations(self):
        data = []
        for userData in self.ratings_matrix.iterrows():
            userId = userData[0]
            ratings = userData[1]
            user_rated_items: pd.Series = ratings[ratings > 0]
            user_rated_items_dict = user_rated_items.to_dict()

            representations_dim = len(PlacesAPI2RepConvertor.get_representation_names())

            rest_profiles_combined = np.zeros(representations_dim)
            weights_total = 0
            for place_id, user_rating in user_rated_items_dict.items():
                restaurant = self.restaurants_data[place_id]
                res_prof = user_rating * np.array(PlacesAPI2RepConvertor.places_api_data_to_vector(restaurant))
                rest_profiles_combined += res_prof
                weights_total += user_rating

            user_profile = (rest_profiles_combined / weights_total)

            user_profile_normed = self._safe_norm(user_profile)

            record = [userId] + user_profile_normed.tolist()
            data.append(record)

        data_header = ["user_name"] + PlacesAPI2RepConvertor.get_representation_names()
        user_representations: pd.DataFrame = pd.DataFrame(data=data, columns=data_header)
        user_representations.fillna(0, inplace=True)
        return user_representations

    def _safe_norm(self, vec_to_normalize) -> np.ndarray:
        norm = np.linalg.norm(vec_to_normalize)
        if norm == 0:
            return np.zeros_like(vec_to_normalize)
        else:
            return vec_to_normalize / np.linalg.norm(vec_to_normalize)

    def get_all_user_representation(self):
        return self.users

    def find_similar(self, user_vector: np.array, top_k: int):
        item_vectors = self.restaurants.iloc[:, 1:].values
        similarities = cosine_similarity(user_vector, item_vectors)
        indices_sorted = np.argsort(similarities[0])
        top_indices = indices_sorted[::-1][:top_k]

        result = self.restaurants.iloc[top_indices].copy()
        result['cos_similarities'] = similarities[0][top_indices]
        result_cleaned = result[['id', 'cos_similarities']].set_index("id")
        return result_cleaned


class SimpleCBModel():

    def __init__(self, data_source: CB_DataSource):
        """Simple CB model build over abstract datasource.

        Args:
            db_data_source (CB_DataSource): _description_
        """
        self.data_source = data_source
        self.all_users: pd.DataFrame = data_source.get_all_user_representation()
        self.users_embeddings = []

    def recommend(self, user_id: int, top_k: int):
        user_vector = self.all_users.iloc[user_id].values[1:].reshape(1, -1)
        res = self.data_source.find_similar(user_vector, top_k)
        return res.reset_index().to_records(index=False).tolist()


    def recommend_by_user_name(self, user_name: str, top_k: int):
        user_row = self.all_users[self.all_users["user_name"] == user_name].iloc[0]
        user_vector = user_row.drop("user_name").values.reshape(1, -1)
        res = self.data_source.find_similar(user_vector, top_k)
        return res.reset_index().to_records(index=False).tolist()


if __name__ == "__main__":
    data_loader = DatasetLoader()
    ratings_matrix = data_loader.load_ratings_matrix()
    print(ratings_matrix)
    data_source = Local_CB_DataSource(1)
    cb_model = SimpleCBModel(data_source)
    recs = cb_model.recommend(42, 5)
    print(recs)
