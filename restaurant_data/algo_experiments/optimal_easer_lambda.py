#!/bin/python3
from dataclasses import dataclass
import os
import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserEvaluation
#from  evaluation_frameworks.general_recommender_evaluation.algorithms.easer_user_based import EaserUserBasedPrecisionEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.baseline import PopularityEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.svd import SVDPrecisionEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.item_knn import ItemItemCFEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.user_knn import UserKnnCFEvaluation
from latex_utils.latex_table_generator import LaTeXTableGenerator, LaTeXTableGeneratorSIUnitx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

# ===================================
# DESCRIPTION
# ===================================
# Finds optimal lambda for the easer algorithm.
# Output: table/plot
#


from utils.config import IMG_OUTPUT_PATH, load_from_pickle, save_to_pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

#region plotting
def plot_result_precision(results):
    # Extract data
    ks = [k for k, _ in results]
    precision = [v['precision@K'] for _, v in results]
    recall = [v['recall@K'] for _, v in results]
    ndcg = [v['ndcg@K'] for _, v in results]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks, precision, marker='^', label='Precision@K')
    #plt.plot(ks, recall, marker='^', label='Recall@K')
    #plt.plot(ks, ndcg, marker='^', label='NDCG@K')

    plt.xlabel('K')
    plt.ylabel('Metric Value')
    plt.title('Evaluation precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "optimal-regularisation-easer_precision.pdf"))
    plt.show()

def plot_result_ndcg(results):
    # Extract data
    ks = [k for k, _ in results]
    #precision = [v['precision@K'] for _, v in results]
    #recall = [v['recall@K'] for _, v in results]
    ndcg = [v['ndcg@K'] for _, v in results]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks, ndcg, marker='^', label='Precision@K')
    #plt.plot(ks, recall, marker='^', label='Recall@K')
    #plt.plot(ks, ndcg, marker='^', label='NDCG@K')

    plt.xlabel('K')
    plt.ylabel('Metric Value')
    plt.title('Evaluation ndcg')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "optimal-regularisation-recall.pdf"))
    plt.show()

def plot_result_recall(results):
    # Extract data
    ks = [k for k, _ in results]
    precision = [v['precision@K'] for _, v in results]
    recall = [v['recall@K'] for _, v in results]
    ndcg = [v['ndcg@K'] for _, v in results]

    # Plot
    plt.figure(figsize=(10, 6))
    #plt.plot(ks, precision, marker='^', label='Precision@K')
    plt.plot(ks, recall, marker='^', label='Recall@K')
    #plt.plot(ks, ndcg, marker='^', label='NDCG@K')

    plt.xlabel('K')
    plt.ylabel('Metric Value')
    plt.title('Evaluation recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "optimal-regularisation-recall.pdf"))
    plt.show()

#endregion

#
# Data load + helpers
#

d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()


#
# Helpers
#

def remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix: pd.DataFrame, min_ratings_count):
    df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) >= min_ratings_count] # drop all users with less than min_ratings_count
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis = 0)] # drop zero columns
    return df_cleaned


#
# Dataclasses
#

@dataclass
class ExperimentResults:
    parameters: Dict
    result: List

def parameter_test(precision_k, regularization_options: List, min_number_of_ratings = 2):
    # get filtered
    filtered_matrix = remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, min_number_of_ratings)

    print("Matrix size:", filtered_matrix.shape)
    user_activity = filtered_matrix.astype(bool).sum(axis=1)
    print("Avg ratings per user:", user_activity.mean())
    density = filtered_matrix.astype(bool).sum().sum() / filtered_matrix.size
    print("Matrix density:", density)

    parameters: dict = {
        "k": precision_k,
        "min_number_of_ratings": min_number_of_ratings,
        "user_activity": user_activity,
        "matrix_density": density,
        "regularization_options": regularization_options
    }

    results = []

    for reg in regularization_options:
        easer_eval = EaserEvaluation(filtered_matrix, precision_k, regularization=reg)
        easer_eval.fit()
        res = easer_eval.evaluate_crossval(20)
        results.append((reg, res))

    return ExperimentResults(parameters, results)


#
# Test params
#

metric_k = 20 # the @k value from evaluation
regularization_options = [_ for _ in range(100, 2000, 100)]
measures: List[ExperimentResults] = []

min_number_of_ratings_options = [4] # leaving users with specific counts of stars

#
# Generate (or load) data
#

load = False
cache_file = "restaurants/hybrid-algorithm/easer/precision_measures.pkl"
if (load):
    measures = load_from_pickle(cache_file, description="restaurant easer precision measures")
else:
    measures = Parallel(n_jobs=1)(
        delayed(parameter_test)(metric_k, regularization_options, min_ratings)
        for min_ratings in min_number_of_ratings_options
    )

    save_to_pickle(measures, cache_file, description="restaurant easer precision measures")


#
# Generate output table
#

def generate_table(measure: ExperimentResults, min_user_rating: int, regularisation_parameters: List[str]):
    datapoints = measure.result
    params = measure.parameters
    print(datapoints)
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

    generator = LaTeXTableGeneratorSIUnitx(
        df,
        column_specs=[(1, 5), (1, 3), (1, 5)],
        column_width=1.5
    )

    print(generator.generate_table(
        caption=f"Přesnost $\\text{{EASE}}^R$ pro různé volby $\\lambda$ pro \\textit{{dataset}} s minimálním počtem hodnocení uživatelů $\\#(r_{{min}}) \ge {min_user_rating}$",
        label="tab:EaserRegMeasure",
        cell_bold_fn=lambda row_idx, col_idx, val: val == df.iloc[:, col_idx].max()
    ))

for i, min_num_rating in enumerate(min_number_of_ratings_options):
    generate_table(measures[i], min_num_rating, regularization_options)
    print("")