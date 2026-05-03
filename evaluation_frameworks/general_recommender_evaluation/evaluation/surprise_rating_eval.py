#!/bin/python3
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import surprise
from scipy.sparse import dok_matrix
from surprise import Dataset, Reader
from surprise.model_selection import KFold


class SurpriseRatingBasedEvaluation(ABC):
    """Base class for Surprise-based recommendation evaluation."""

    def __init__(self, rating_matrix: pd.DataFrame, test_size: float = 0.2, rating_scale=(1, 5)):
        self.rating_matrix = rating_matrix

        self.test_size = test_size
        self.rating_scale = rating_scale

        self.trainset = None
        self.testset = None
        self.train_matrix = None
        self.algo = None

        self._initiate_train_matrix()

    @staticmethod
    def _matrix_to_ratings_df(rating_matrix: pd.DataFrame) -> pd.DataFrame:
        """Long-form (user, item, rating) ze sparse matice; zachová raw index/sloupce (např. jména uživatelů)."""
        stacked = rating_matrix.stack()
        stacked = stacked[(stacked != 0) & stacked.notna()]
        out = stacked.reset_index()
        out.columns = ["user", "item", "rating"]
        return out

    @abstractmethod
    def fit(self):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self):
        """Run evaluation and return results."""
        pass

    def evaluate_crossval(self, n_splits=5):
        ratings = self._matrix_to_ratings_df(self.rating_matrix)
        if ratings.shape[0] == 0:
            raise ValueError(
                "Žádná hodnocení pro křížovou validaci (prázdná matice po filtru nebo mimo rating_scale)."
            )

        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(ratings[["user", "item", "rating"]], reader)

        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        results = []

        for trainset, testset in kf.split(data):
            self.trainset = trainset
            self.testset = testset
            self._initiate_train_matrix_from_existing_trainset()
            self.fit()
            results.append(self.evaluate())

        return self._average_results(results)

    def _initiate_train_matrix(self):
        """Initiates matrix of train data."""
        self._initiate_test_sets()
        self._initiate_train_matrix_from_existing_trainset()

    def _initiate_train_matrix_from_existing_trainset(self):
        """Initiates matrix of train data using already created test sets."""
        num_users = self.trainset.n_users
        num_items = self.trainset.n_items
        self.train_matrix = dok_matrix((num_users, num_items))

        for uid, iid, rating in self.trainset.all_ratings():
            self.train_matrix[int(uid), int(iid)] = rating

    def _average_results(self, results):
        """Averages results from the cross validation."""
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        return avg

    def _initiate_test_sets(self) -> tuple:
        """Creates Surprise train/test split from rating matrix."""
        ratings = self._matrix_to_ratings_df(self.rating_matrix)
        if ratings.shape[0] == 0:
            raise ValueError(
                "Žádná ne-nulová hodnocení pro Surprise (prázdná matice po filtru nebo špatný rating_scale). "
                "Zkontroluj dataset/restaurants/places.json."
            )
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(ratings[["user", "item", "rating"]], reader)

        self.trainset, self.testset = surprise.model_selection.train_test_split(data, test_size=self.test_size)
        return self.trainset, self.testset
