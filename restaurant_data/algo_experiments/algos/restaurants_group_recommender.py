from abc import ABC, abstractmethod
import numpy as np
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer_cached import EaserCached
from evaluation_frameworks.general_recommender_evaluation.algorithms.user_knn import UserKnnCFEvaluation
from restaurant_data.algo_experiments.algos.cb_model import SimpleCBModel
from typing import List, Dict


class GroupRecommender(ABC):
    def __init__(self):
        pass

class MoviesGroupRecommender:

    def __init__(self, easer: EaserCached):
        self.easer = easer
        pass

class RestaurantGroupRecommender:

    def __init__(self, knn_model: UserKnnCFEvaluation, cb_model: SimpleCBModel, alpha_param: float = 0.2):
        """ Group recommender for restaurants.

            Using combination of knn_model and cb_model.

        Args:
            knn_model (UserKnnCFEvaluation): _description_
            cb_model (SimpleCBModel): _description_
            alpha_param (float, optional): _description_. Defaults to 0.2.
        """
        self.alpha = alpha_param
