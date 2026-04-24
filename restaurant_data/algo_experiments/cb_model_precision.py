import numpy as np
import pandas as pd
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx
from restaurant_data.algo_experiments.algos.cb_model import Local_CB_DataSource, SimpleCBModel
from restaurant_data.algo_experiments.algos.cb_evaluation import CBModelEvaluation
from dataset.data_access import DatasetLoader

dataset = DatasetLoader()
rating_matrix = dataset.load_ratings_matrix()

min_user_rating = 4 # We use all the users from the test set (we leave this optional for experiments)
rating_matrix = DatasetLoader.remove_users_from_ratings_matrix_by_ratings_count(rating_matrix, min_user_rating)
data_source = Local_CB_DataSource(min_user_rating)

cb_model = SimpleCBModel(data_source)
metric_k = 20
evaluator = CBModelEvaluation(rating_matrix, cb_model, k=metric_k)
results = evaluator.evaluate()

precision = results[f"precision@{metric_k}"]
recall = results[f"recall@{metric_k}"]
ndcg = results[f"ndcg@{metric_k}"]

data = {
    "Algorithm": ["Simple CB"],
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
    #cell_bold_fn=lambda row_idx, col_idx, val: val == df.iloc[:, col_idx].max()
))

print("")
