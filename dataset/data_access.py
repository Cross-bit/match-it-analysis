#!/bin/python3

import os
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy.sparse import csr_matrix

from surprise import Dataset, Reader
from utils.config import MOVIES_DATASET_ROOT, RESTAURANT_DATASET_ROOT


class MovieLensDatasetLoader:
    def __init__(self, dataset_dir: str = "ml-1m", movies_file = "movies.csv", ratings_file="ratings.csv"):
        """
        Initialize the MovieLens dataset loader.

        Args:
            dataset_dir (str): Path to the MovieLens dataset directory (e.g., "ml-1m", "ml-25m").
        """
        self.dataset_dir_name = dataset_dir
        self.dataset_dir = str(self._resolve_dataset_dir(dataset_dir))
        self.movies_file = os.path.join(self.dataset_dir, movies_file) if movies_file != "" else ""
        self.ratings_file = os.path.join(self.dataset_dir, ratings_file) if ratings_file != "" else ""

        self.movies_df = None
        self.ratings_df = None

    @staticmethod
    def _resolve_dataset_dir(dataset_dir: str) -> Path:
        dataset_path = MOVIES_DATASET_ROOT / dataset_dir
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        return dataset_path

    def load_data(self, fill_zeroes = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the MovieLens dataset (movies and ratings) as a full user-item rating matrix (dense format).

        WARNING: This creates a large pivot table (userId x movieId), which can consume significant memory.
        Recommended only for smaller datasets (e.g., ml-100k, ml-1m). Avoid with ml-25m, ml-32m, etc.

        Args:
            fill_zeroes (float): Fill NaNs with zeroes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (movies dataframe, ratings matrix)
        """
        if self.movies_file != "":
            self.movies_df = pd.read_csv(
                self.movies_file,
                names=["movieId", "title", "genres"],
                encoding="latin1"
            )

        if self.ratings_file != "":
            raw_ratings_df = pd.read_csv(
                self.ratings_file,
                dtype={
                    "userId": int,
                    "movieId": int,
                    "rating": float,
                    "timestamp": int
                },
                low_memory=False
            ).drop("timestamp", axis=1)

            self.ratings_df = raw_ratings_df.pivot(index="userId", columns="movieId", values="rating")

            self.ratings_df = self.ratings_df.fillna(0) if (fill_zeroes) else self.ratings_df

        return self.movies_df, self.ratings_df

    def load_sparse_ratings(self, use_cache=False) -> Tuple[pd.DataFrame, csr_matrix, dict, dict]:
        """
        Loads the MovieLens dataset and returns a sparse user-item rating matrix.
        Suitable for large datasets like ml-25m or ml-32m.
        Args:
            use_cache (bool): If true, loads/stores data from/to pickles. Uses dataset_dir from ctor for the file .pkl file name.

        Returns:
            movies_df: Movie metadata
            ratings_sparse: scipy.sparse.csr_matrix (users x movies)
            user_id_map: map from sparse row index → real userId
            movie_id_map: map from sparse col index → real movieId
        """

        pickle_path = os.path.join(self.dataset_dir, f"sparse-cached-{self.dataset_dir_name}.pkl")

        if use_cache and os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                print("✅ Loaded sparse matrix from pickle.")
                return pickle.load(f)

        print("Loading data from the .csv ...")

        if self.movies_file != "":
            self.movies_df = pd.read_csv(
                self.movies_file,
                names=["movieId", "title", "genres"],
                encoding="latin1",
                header=0
            )

        raw_ratings_df = pd.read_csv(
            self.ratings_file,
            dtype={
                "userId": int,
                "movieId": int,
                "rating": float,
                "timestamp": int
            },
            low_memory=False,
            header=0
        ).drop("timestamp", axis=1)

        print("Computing sparse representation...")

        # Encode user and movie IDs as categorical codes
        user_cats = raw_ratings_df["userId"].astype("category")
        movie_cats = raw_ratings_df["movieId"].astype("category")

        raw_ratings_df["user_idx"] = user_cats.cat.codes
        raw_ratings_df["movie_idx"] = movie_cats.cat.codes

        user_id_map = dict(enumerate(user_cats.cat.categories))
        movie_id_map = dict(enumerate(movie_cats.cat.categories))

        ratings_sparse = csr_matrix((
            raw_ratings_df["rating"],
            (raw_ratings_df["user_idx"], raw_ratings_df["movie_idx"])
        ))

        result = (self.movies_df, ratings_sparse, user_id_map, movie_id_map)

        if use_cache:
            with open(pickle_path, "wb") as f:
                pickle.dump(result, f)
                print("💾 Saved sparse matrix to pickle.")

        return result

    def filter_csr_by_interaction_thresholds(
        self,
        ratings_csr: csr_matrix,
        user_id_map: Dict[int, int],
        item_id_map: Dict[int, int],
        min_user_interactions: int = 50,
        min_item_interactions: int = 20,
        rating_threshold: float = 4.0,
    ) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
        """
        Filters the CSR matrix and id maps by user/item interaction thresholds.

        Returns:
            - filtered_csr
            - new_user_id_map: row index → real userId
            - new_item_id_map: col index → real movieId
        """

        # 1. Remove zero values and mark positive values
        filtered = ratings_csr.copy()
        filtered.data = np.where(filtered.data >= rating_threshold, 1, 0)
        filtered.eliminate_zeros() # remove zeroes

        # 2. Sum all user and items interactions
        user_inter_counts = np.array(filtered.sum(axis=1)).flatten()
        item_inter_counts = np.array(filtered.sum(axis=0)).flatten()

        # find indices of users and items with sufficient counts of interactions
        valid_user_indices = np.where(user_inter_counts >= min_user_interactions)[0]
        valid_item_indices = np.where(item_inter_counts >= min_item_interactions)[0]

        # 3. Filter the valud data
        filtered_csr = ratings_csr[valid_user_indices, :][:, valid_item_indices]

        new_user_id_map = {
            new_idx: user_id_map[old_idx]
            for new_idx, old_idx in enumerate(valid_user_indices)
        }

        new_item_id_map = {
            new_idx: item_id_map[old_idx]
            for new_idx, old_idx in enumerate(valid_item_indices)
        }

        return filtered_csr, new_user_id_map, new_item_id_map


    def split_csr_train_val_test(
        self,
        csr: csr_matrix,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        """
        Rozdělí CSR matici na train, val, test podle interakcí jednotlivých uživatelů.
        Každý uživatel zůstává v maticích, i pokud má v některé z nich 0 interakcí.
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"

        np.random.seed(seed)
        num_users, num_items = csr.shape

        train_data = []
        val_data = []
        test_data = []

        for user_id in range(num_users):
            start_ptr, end_ptr = csr.indptr[user_id], csr.indptr[user_id + 1]
            item_indices = csr.indices[start_ptr:end_ptr]
            item_data = csr.data[start_ptr:end_ptr]

            if len(item_indices) == 0:
                continue  # uživatel bez interakcí

            idx = np.arange(len(item_indices))
            np.random.shuffle(idx)

            train_end = int(len(idx) * train_ratio)
            val_end = train_end + int(len(idx) * val_ratio)

            train_idx = idx[:train_end]
            val_idx = idx[train_end:val_end]
            test_idx = idx[val_end:]

            if len(train_idx) > 0:
                train_data.append((user_id, item_indices[train_idx], item_data[train_idx]))
            if len(val_idx) > 0:
                val_data.append((user_id, item_indices[val_idx], item_data[val_idx]))
            if len(test_idx) > 0:
                test_data.append((user_id, item_indices[test_idx], item_data[test_idx]))

        def build_csr(data, num_users, num_items):
            data_vals = []
            row_ind = []
            col_ind = []

            for user_id, items, values in data:
                data_vals.extend(values)
                col_ind.extend(items)
                row_ind.extend([user_id] * len(items))

            return csr_matrix((data_vals, (row_ind, col_ind)), shape=(num_users, num_items))

        train_csr = build_csr(train_data, num_users, num_items)
        val_csr = build_csr(val_data, num_users, num_items)
        test_csr = build_csr(test_data, num_users, num_items)

        return train_csr, val_csr, test_csr

    def split_csr_by_users_full_mapping(
        self,
        csr: csr_matrix,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    ) -> Tuple[
        Tuple[csr_matrix, Dict[int, int], Dict[int, int]],
        Tuple[csr_matrix, Dict[int, int], Dict[int, int]],
        Tuple[csr_matrix, Dict[int, int], Dict[int, int]]
    ]:
        """
        Rozdělí csr matici podle uživatelů a vrátí plnohodnotné CSR matice s mapami původních ID.

        Returns:
            (train_csr, train_user_map, train_item_map),
            (val_csr, val_user_map, val_item_map),
            (test_csr, test_user_map, test_item_map)
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"

        np.random.seed(seed)
        n_users, n_items = csr.shape

        user_indices = np.arange(n_users)
        np.random.shuffle(user_indices)

        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)

        train_users = user_indices[:n_train]
        val_users = user_indices[n_train:n_train + n_val]
        test_users = user_indices[n_train + n_val:]

        def subset(users: np.ndarray):
            sub_csr = csr[users, :]
            used_items = np.unique(sub_csr.indices)

            # mapování uživatelů a položek
            user_id_map = {i: orig_id for i, orig_id in enumerate(users)}
            item_id_map = {i: orig_id for i, orig_id in enumerate(used_items)}

            # převod na zmenšenou matici
            rows, cols = sub_csr.nonzero()
            data = sub_csr.data
            new_rows = rows
            new_cols = np.array([np.where(used_items == col)[0][0] for col in cols])

            reduced_csr = csr_matrix((data, (new_rows, new_cols)), shape=(len(users), len(used_items)))

            return reduced_csr, user_id_map, item_id_map

        train_csr, train_user_map, train_item_map = subset(train_users)
        val_csr, val_user_map, val_item_map = subset(val_users)
        test_csr, test_user_map, test_item_map = subset(test_users)

        return (
            (train_csr, train_user_map, train_item_map),
            (val_csr, val_user_map, val_item_map),
            (test_csr, test_user_map, test_item_map)
        )

    def filter_by_popularity(self, pop_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters the dataset to include only the top X% most rated movies.

        Args:
            pop_threshold (float): Top percentage of most-rated movies to keep (0-100).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (filtered movies dataframe, filtered ratings matrix)
        """
        movie_popularity = self.ratings_df.notna().sum(axis=0)
        valid_count = int(movie_popularity.size * (pop_threshold / 100))

        top_movies = movie_popularity.sort_values(ascending=False).iloc[:valid_count].index

        filtered_movies_df = self.movies_df[self.movies_df["movieId"].isin(top_movies)]
        filtered_ratings_df = self.ratings_df[top_movies]

        return filtered_movies_df, filtered_ratings_df

    def user_genre_avg_ratings(self) -> pd.DataFrame:
        """
        Computes the average rating per user for each genre.

        Returns:
            pd.DataFrame: User-genre average ratings.
        """
        ratings_long = self.ratings_df.reset_index().melt(id_vars="userId", var_name="movieId", value_name="rating")
        ratings_long.dropna(subset=["rating"], inplace=True)

        movie_genres = self.movies_df[["movieId", "genres"]]
        merged = ratings_long.merge(movie_genres, on="movieId", how="left")

        genres_dummies = merged["genres"].str.get_dummies(sep="|")
        merged = pd.concat([merged, genres_dummies], axis=1)

        for genre in genres_dummies.columns:
            merged[genre] = merged[genre] * merged["rating"]

        genre_sum = merged.groupby("userId")[genres_dummies.columns].sum()
        genre_count = merged[genres_dummies.columns].astype(bool).groupby(merged["userId"]).sum()

        genre_avg = genre_sum / genre_count.replace(0, pd.NA)
        return genre_avg

    def get_surprise_trainset(self):
        """
        Converts the ratings matrix to a Surprise Trainset object.

        Returns:
            surprise.Trainset: Full training set ready for Surprise algorithms.
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data is not loaded. Call load_data() first.")

        long_df = self.ratings_df.stack().reset_index()
        long_df.columns = ['userID', 'itemID', 'rating']

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(long_df, reader)
        return data.build_full_trainset()


DATASET_FILE_NAME = "places.json"


class DatasetLoader:
    def __init__(self, file_name=DATASET_FILE_NAME):
        self.DATASET_DATA_FILE = str(self._resolve_dataset_file(file_name))

    @staticmethod
    def _resolve_dataset_file(file_name: str) -> Path:
        dataset_file = RESTAURANT_DATASET_ROOT / file_name
        if not dataset_file.exists():
            raise FileNotFoundError(f"Restaurant dataset file not found: {dataset_file}")
        return dataset_file

    def load_ratings_matrix(self, data_filename="") -> pd.DataFrame:
        data_filename = self.DATASET_DATA_FILE if (data_filename == "") else data_filename
        raw_data: Dict = self.load_data_raw(data_filename)

        places = raw_data["places"]
        all_places_ids = [item["id"] for item in places]
        all_user_ratings = self._get_unique_users_with_ratings(places)
        all_users_names = all_user_ratings.keys()

        df = pd.DataFrame(0, index=all_users_names, columns=all_places_ids)

        for user_name in all_users_names:
            places_ratings = all_user_ratings[user_name]
            for place_rating in places_ratings:
                place_id, rating = place_rating
                df.loc[user_name, place_id] = rating

        return df

    @staticmethod
    def remove_users_from_ratings_matrix_by_ratings_count(
        ratings_matrix: pd.DataFrame, min_ratings_count: int, drop_zero_cols=True
    ) -> pd.DataFrame:
        df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) >= min_ratings_count]
        if drop_zero_cols:
            df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
        return df_cleaned

    def load_data_raw(self, data_filename=""):
        data_filename = self.DATASET_DATA_FILE if (data_filename == "") else data_filename
        return self._load_places_data(data_filename)

    def _load_places_data(self, db_file_name):
        with open(db_file_name, "r", encoding="utf-8") as file:
            return json.load(file)

    def _get_unique_users_with_ratings(self, places_data: Dict) -> Dict[str, List[Tuple[str, int]]]:
        user_ratings = {}

        for p in places_data:
            if "reviews" in p and p["reviews"] != []:
                reviews = p["reviews"]
                for review in reviews:
                    author = review["authorAttribution"]["displayName"]
                    if not user_ratings.get(author):
                        user_ratings[author] = []
                    user_ratings[author].append((p["id"], review["rating"]))

        return user_ratings





##TEST functions

#def validate_csr_filtering_vectorized(
#    original_csr: csr_matrix,
#    filtered_csr: csr_matrix,
#    new_user_id_map: dict,
#    new_item_id_map: dict,
#    original_user_id_map: dict,
#    original_item_id_map: dict,
#):
#    user_id_to_old_idx = {v: k for k, v in original_user_id_map.items()}
#    item_id_to_old_idx = {v: k for k, v in original_item_id_map.items()}
#
#    f_coo = filtered_csr.tocoo()
#
#    errors = 0
#    for i, j, val_filtered in zip(f_coo.row, f_coo.col, f_coo.data):
#        user_id = new_user_id_map[i]
#        item_id = new_item_id_map[j]
#        old_i = user_id_to_old_idx[user_id]
#        old_j = item_id_to_old_idx[item_id]
#
#        val_original = original_csr[old_i, old_j]
#        if val_filtered != val_original:
#            print(f"❌ Mismatch at filtered({i},{j}) → user_id={user_id}, item_id={item_id}")
#            print(f"    filtered_val={val_filtered} vs original_val={val_original}")
#            errors += 1
#
#    if errors == 0:
#        print("✅ Validation passed: all values match.")
#    else:
#        print(f"⚠️ {errors} mismatches found.")