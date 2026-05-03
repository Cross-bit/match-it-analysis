#!/bin/python3
import os
import sys
from typing import Any
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from evaluation_frameworks.general_recommender_evaluation.algorithms.item_knn import ItemItemCFEvaluation
from latex_utils.latex_table_generator import LaTeXTableGenerator, LaTeXTableGeneratorSIUnitx
from latex_utils.latex_multihead_generator import MultiHeaderLaTeXTableGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *
from utils.config import load_from_pickle, save_to_pickle

# ===================================
# !!!!! DESCRIPTION -- NOTICE!!!!!
# ===================================
# Attempt to find optimal k for KNN for initial dataset - infeasible due to sparsity.
# Average number of neighbors is ~10.
#

d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()

#
# Helpers
#

def remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix: pd.DataFrame, min_ratings_count):
    df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) > min_ratings_count] # drop all users with less than min_ratings_count
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis = 0)] # drop zero columns
    return df_cleaned

#
# Dataclass
#

@dataclass
class ExperimentResults:
    parameters: Dict
    result: List

def parameter_test(precision_k, knn_k_options: List, min_number_of_ratings = 2):

    # get filtered
    filtered_matrix = remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, min_number_of_ratings)

    print("Matrix size:", filtered_matrix.shape)
    user_activity = filtered_matrix.astype(bool).sum(axis=1)
    print("Avg ratings per user:", user_activity.mean())
    density = filtered_matrix.astype(bool).sum().sum() / filtered_matrix.size
    print("Matrix density:", density)

    parameters: dict = {
        "precision_k": precision_k,
        "knn_k": knn_k_options,
        "min_number_of_ratings": min_number_of_ratings,
        "user_activity": user_activity,
        "matrix_density": density,
    }

    results = []

    for knn_k_value in knn_k_options:
        easer_eval = ItemItemCFEvaluation(filtered_matrix, precision_k, knn_k_value)
        easer_eval.fit()
        res = easer_eval.evaluate_crossval(20)
        results.append((knn_k_value, res))

    return ExperimentResults(parameters, results)


#
# Test params
#

metric_k = 20 # the @k value from evaluation
knn_k_values_options = [_ for _ in range(10, 100, 10)]
measures: List[ExperimentResults] = []

min_number_of_ratings_options = [3] # leaving users with specific counts of stars

#
# Generate (or load) data
#

load = False
cache_file = "restaurants/hybrid-algorithm/item-knn/optimal_item_item_k_data.pkl"
if (load):
    measures = load_from_pickle(cache_file, description="restaurant item-knn sweep")
else:
    measures = Parallel(n_jobs=2)(
        delayed(parameter_test)(metric_k, knn_k_values_options, min_ratings)
        for min_ratings in min_number_of_ratings_options
    )

    save_to_pickle(measures, cache_file, description="restaurant item-knn sweep")

#
# Generate output table
#

def generate_table(measure: ExperimentResults, min_user_rating: int, regularisation_parameters: List[str]):
    datapoints = measure.result
    params = measure.parameters
    precision = [v['precision@K'] for _, v in datapoints]
    recall = [v['recall@K'] for _, v in datapoints]
    ndcg = [v['ndcg@K'] for _, v in datapoints]


    data = {
        "$\\lambda$": [str(_) for _ in regularisation_parameters],
        f"precision@{metric_k}": np.round(precision, 5),
        f"recall@{metric_k}": np.round(recall, 3),
        f"NDCG@{metric_k}": np.round(ndcg, 3)
    }

    df = pd.DataFrame(data)

    generator = LaTeXTableGenerator(
        df,
        column_specs=[(1, 5), (1, 3), (1, 5)],
        column_width=1.5
    )

    print(generator.generate_table(
        caption=f"Přesnost $\\text{{EASE}}^R$ pro různé volby $\\lambda$ pro \\textit{{dataset}} s minimálním počtem hodnocení pro uživatele $\\#(r_{{min}}) \\ge {min_user_rating}$",
        label="tab:EaserRegMeasure",
    ))

    print("")

for i, min_num_rating in enumerate(min_number_of_ratings_options):
    generate_table(measures[i], min_num_rating, knn_k_values_options)

