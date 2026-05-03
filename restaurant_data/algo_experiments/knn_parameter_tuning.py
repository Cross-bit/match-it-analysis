import pickle
import numpy as np
import pandas as pd
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx
from dataset.data_access import DatasetLoader
from evaluation_frameworks.general_recommender_evaluation.algorithms.item_knn import ItemKnnCFEvaluation

# =========================
# DESCRIPTIONS
# =========================
# In this file we conduct hyperparameter tuning
# for the item-KNN.
# (On Prague test dataset we have on average 10 neighbors)
#


dataset = DatasetLoader()
rating_matrix = dataset.load_ratings_matrix()
min_user_ratings = 4
ratings_filtered = DatasetLoader.remove_users_from_ratings_matrix_by_ratings_count(rating_matrix, min_user_ratings)
metric_k = 20
k_values = range(0, 15)

eval_results = []
for k in k_values:
    evaluation = ItemKnnCFEvaluation(ratings_filtered, metric_k, k)
    result = evaluation.evaluate_crossval(20)
    eval_results.append(result)


#
# ---- PLOTTING -----
#

precision = [v["precision@K"] for v in eval_results]
recall = [v["recall@K"] for v in eval_results]
ndcg = [v["ndcg@K"] for v in eval_results]

data = {
    "Algorithm": list(k_values),
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
    caption=f"Porovnání jednotlivých algoritmů na \\textit{{datasetu}} restaurací pro $\\#(r_{{min}}) \\ge {min_user_ratings}$.",
    label="tab:AlgosComparisionTextTable",
    cell_bold_fn=lambda row_idx, col_idx, val: val == df.iloc[:, col_idx].max()
))

