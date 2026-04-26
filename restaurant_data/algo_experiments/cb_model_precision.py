import numpy as np
import pandas as pd
import argparse
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx
from restaurant_data.algo_experiments.algos.cb_model import Local_CB_DataSource, SimpleCBModel
from restaurant_data.algo_experiments.algos.cb_evaluation import CBModelEvaluation
from dataset.data_access import DatasetLoader
from utils.config import load_from_pickle, save_to_pickle

parser = argparse.ArgumentParser()
parser.add_argument("--min-user-rating", type=int, default=4)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--train-ratio", type=float, default=0.8)
parser.add_argument("--mode", choices=["auto", "load", "compute"], default="auto")
args = parser.parse_args()

dataset = DatasetLoader()
rating_matrix = dataset.load_ratings_matrix()

min_user_rating = args.min_user_rating
rating_matrix = DatasetLoader.remove_users_from_ratings_matrix_by_ratings_count(rating_matrix, min_user_rating)
data_source = Local_CB_DataSource(min_user_rating)

cb_model = SimpleCBModel(data_source)
metric_k = args.k
evaluator = CBModelEvaluation(rating_matrix, cb_model, k=metric_k, train_ratio=args.train_ratio)
cache_key = (
    f"restaurants/hybrid-algorithm/cb/simple_cb_rmin_{min_user_rating}"
    f"_k_{metric_k}_tr_{args.train_ratio:.2f}.pkl"
)
description = (
    f"restaurant simple CB metrics r_min={min_user_rating}, "
    f"k={metric_k}, train_ratio={args.train_ratio:.2f}"
)
if args.mode == "load":
    results = load_from_pickle(cache_key, description=description)
elif args.mode == "compute":
    results = evaluator.evaluate()
    save_to_pickle(results, cache_key, description=description)
else:
    try:
        results = load_from_pickle(cache_key, description=description)
    except FileNotFoundError:
        results = evaluator.evaluate()
        save_to_pickle(results, cache_key, description=description)

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
))

print("")
