import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, KNNBasic, Reader
from utils.config import IMG_OUTPUT_PATH
from utils.config import load_from_pickle, save_to_pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()

#region Plotting

def plot_comparison_with_labels(results_user_knn: List[float], results_item_knn: List[float], x_range: range, title: str = "Porovnání User KNN a Item KNN"):
    """
    Plots line charts of results_user_knn and results_item_knn with data point labels.
    """
    x = list(x_range)
    axis_label_size = 22
    y_axis_label_size = 22
    tick_label_size = 18
    value_label_size = 10
    legend_size = 16
    title_size = 24

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.5), 6), dpi=100)

    # Plot user_kNN line
    ax.plot(x, results_user_knn, marker='o', label='Uživatelové KNN', color='blue')
    for i, val in enumerate(results_user_knn):
        x_offset = 0.35 if i == 0 else 0.0
        y_offset = 1.0 if i == 0 else 0.5
        ax.text(x[i] + x_offset, val + y_offset, f'{val:.2f}', ha='center', va='bottom', fontsize=value_label_size)

    # Plot item_kNN line
    ax.plot(x, results_item_knn, marker='s', label='Položkové KNN', color='green')
    for i, val in enumerate(results_item_knn):
        x_offset = -0.35 if i == 0 else 0.0
        y_offset = -1.0 if i == 0 else -0.5
        ax.text(x[i] + x_offset, val + y_offset, f'{val:.2f}', ha='center', va='top', fontsize=value_label_size)

    ax.set_xlabel("Minimální počet hodnocení uživatelů", fontsize=axis_label_size)
    ax.set_ylabel("Průměrný počet sousedů", fontsize=y_axis_label_size)
    ax.set_title(title, fontsize=title_size)
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    ax.legend(fontsize=legend_size)
    ax.grid(True)

    plt.tight_layout()
    # Optional save
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "knn-neighbors-avg-count.pdf"))
    plt.show()

#endRegion

#
# Helpers
#

def get_surprise_trainset(ratings_matrix_df):
    long_df = ratings_matrix_df.stack().reset_index()
    long_df.columns = ['userID', 'itemID', 'rating']

    reader = Reader(rating_scale=(1, 5))  # Adjust scale as needed
    data = Dataset.load_from_df(long_df, reader)
    trainset = data.build_full_trainset()
    return trainset


def remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix: pd.DataFrame, min_ratings_count):
    df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) >= min_ratings_count] # drop all users with less than min_ratings_count
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis = 0)] # drop zero columns
    return df_cleaned

def get_average_test(dataset: Dataset, user_knn: bool, kneighbors_k = 1):

    sim_options = {
        'name': 'cosine',
        'user_based': user_knn
    }

    algo = KNNBasic(sim_options=sim_options, k = kneighbors_k)
    algo.fit(dataset)
    avg_neighbors = np.mean([np.count_nonzero(algo.sim[i]) for i in range(len(algo.sim))])

    return avg_neighbors

def run_test(min_number_ratings, user_knn = True, runs_count = 1):
    filtered_dataset = remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, min_number_ratings)
    trainset = get_surprise_trainset(filtered_dataset)

    if runs_count <= 1:
        return get_average_test(trainset, user_knn)

    tmp_res = []
    for i in range(runs_count):
        tmp_res.append(get_average_test(trainset, user_knn))

    return np.average(tmp_res)

def run_new_evaluation(x_range: range):
    min_number_ratings_range = x_range

    results_user_knn = []
    results_item_knn = []

    for i in min_number_ratings_range:
        print(f"Running user KNN {i}")
        res_user_knn = run_test(i, True)
        print(f"Running item KNN {i}")
        res_item_knn = run_test(i, False)
        results_user_knn.append(res_user_knn)
        results_item_knn.append(res_item_knn)

    save_to_pickle(
        (results_user_knn, results_item_knn),
        "restaurants/hybrid-algorithm/knn/avg_neighbor_knn.pkl",
        description="restaurant knn neighbors profile",
    )

    return results_user_knn, results_item_knn

def get_results_with_cache(x_range: range, mode: str):
    cache_key = "restaurants/hybrid-algorithm/knn/avg_neighbor_knn.pkl"
    description = "restaurant knn neighbors profile"

    if mode == "load":
        return load_from_pickle(cache_key, description=description)

    if mode == "compute":
        return run_new_evaluation(x_range)

    # auto mode: load if possible, otherwise compute and save
    try:
        return load_from_pickle(cache_key, description=description)
    except FileNotFoundError:
        print("Cache not found, computing fresh results...")
        return run_new_evaluation(x_range)



# =================================
#
# Execution
#
# =================================



results_user_knn = []
results_item_knn = []

x_range = range(2, 31)
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["auto", "load", "compute"], default="auto")
args = parser.parse_args()
results_user_knn, results_item_knn = get_results_with_cache(x_range, args.mode)


plot_comparison_with_labels(results_user_knn, results_item_knn, x_range)