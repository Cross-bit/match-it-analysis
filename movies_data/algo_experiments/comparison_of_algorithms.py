#!/bin/python3
import os
import sys
import numpy as np
import pandas as pd
from  evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserEvaluation
from joblib import Parallel, delayed

#from  evaluation_frameworks.general_recommender_evaluation.algorithms.easer_user_based import EaserUserBasedPrecisionEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.baseline import PopularityEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.svd import SVDPrecisionEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.item_knn import ItemItemCFEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.user_knn import UserKnnCFEvaluation
from latex_utils.latex_table_generator import LaTeXTableGenerator, LaTeXTableGeneratorSIUnitx


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access  import *

# ===================================
# DESCRIPTION
# ===================================
# Tests of the easer algorithm on the places maps data.
#

# ===========================
# Surprise evaluation
# ===========================
#
#
#

d_loader = MovieLensDatasetLoader()
_, ratings_matrix = d_loader.load_data(True)


def remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix: pd.DataFrame, min_ratings_count):
    df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) > min_ratings_count] # drop all users with less than min_ratings_count
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis = 0)] # drop zero columns
    return df_cleaned

min_user_rating = 3

filtered_matrix = ratings_matrix # remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, min_user_rating)

print(filtered_matrix)
print("Matrix size:", filtered_matrix.shape)

user_activity = filtered_matrix.astype(bool).sum(axis=1)
print("Avg ratings per user:", user_activity.mean())

density = filtered_matrix.astype(bool).sum().sum() / filtered_matrix.size
print("Matrix density:", density)

metric_k = 20
test_size = 0.2
#easer_u_eval = EaserUserBasedPrecisionEvaluation(filtered_matrix, k, regularization=500)

evaluations = {
    "$\\text{EASE}^R$": EaserEvaluation(filtered_matrix, metric_k, test_size=test_size, regularization=700),
    "Popularity": PopularityEvaluation(filtered_matrix, metric_k, test_size=test_size),
    #"Item-Item": ItemItemCFEvaluation(filtered_matrix,k=metric_k, algorithm_k=70, test_size=test_size),
    #"UserKNN": UserUserCFEvaluation(filtered_matrix, k=metric_k, algorithm_k=30, test_size=test_size),
    "SVD": SVDPrecisionEvaluation(filtered_matrix, metric_k, test_size=test_size, n_factors=50),
}

#results = {}
#for eval_name, eval in evaluations.items():
#    eval.fit()
#    res = eval.evaluate_crossval(20)
#    results[eval_name] = res

def run_eval(eval_name, eval):
    eval.fit()
    return (eval_name, eval.evaluate_crossval(2))

results_list = Parallel(n_jobs=-1)(delayed(run_eval)(name, ev) for name, ev in evaluations.items())
results = dict(results_list)


precision = [v["precision@K"] for v in results.values()]
recall = [v["recall@K"] for v in results.values()]
ndcg = [v["ndcg@K"] for v in results.values()]

data = {
    "Algorithm": results.keys(),
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
    caption=f"Porovnání jednotlivých algoritmů na \\textit{{datasetu}} restaurací pro $\\#(r_{{min}}) \ge {min_user_rating}$.",
    label="tab:AlgosComparisionTextTable",
    cell_bold_fn=lambda row_idx, col_idx, val: val == df.iloc[:, col_idx].max()
))

print("")

exit()

#print(f"pop: {popularity_eval.evaluate_crossval(10)}")
#print(f"svd: {svd_eval.evaluate_crossval(10)}")
#print(f"easer: {easer_eval.evaluate_crossval(10)}")


#print(f"easer_u: {easer_u_eval.evaluate_crossval(10)}")


#
#eb = EaserBase()
#G = eb.find_coocurance_matrix(ratings_matrix)
#B = pd.DataFrame(data = eb.fit(G))
#
#
#top_connections = B[123].sort_values(ascending=False).head(10)
#nonzeros_per_row = (np.abs(B.values) > 1e-4).sum(axis=1)
#optimal_regularisation = [1, 10, 100, 500, 1000, 2000, 5000, 10000]

#print(nonzeros_per_row.mean())
#print(top_connections)