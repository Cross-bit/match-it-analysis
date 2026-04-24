import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile
from surprise import Dataset, KNNBasic, Reader
from utils.config import IMG_OUTPUT_PATH
from utils.config import load_from_pickle, save_to_pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()

#region Plotting

def plot_comparison_with_labels(results_user_knn: List[float], results_item_knn: List[float], x_range: range, title: str = "User vs Item KNN Comparison"):
    """
    Plots line charts of results_user_knn and results_item_knn with data point labels.
    """
    x = list(x_range)

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.5), 6), dpi=100)

    # Plot user_kNN line
    ax.plot(x, results_user_knn, marker='o', label='User KNN', color='blue')
    for i, val in enumerate(results_user_knn):
        ax.text(x[i], val + 0.5, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot item_kNN line
    ax.plot(x, results_item_knn, marker='s', label='Item KNN', color='green')
    for i, val in enumerate(results_item_knn):
        ax.text(x[i], val - 0.5, f'{val:.2f}', ha='center', va='top', fontsize=8)

    ax.set_xlabel("Minimal number of user ratings")
    ax.set_ylabel("Average number of neighbors")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
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

def run_test(min_number_ratings, user_knn = True, runs_count = 3):
    filtered_dataset = remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, min_number_ratings)
    trainset = get_surprise_trainset(filtered_dataset)

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



# =================================
#
# Execution
#
# =================================



results_user_knn = []
results_item_knn = []

x_range = range(2, 31)
load = False
if (load):
    results_user_knn, results_item_knn = load_from_pickle(
        "restaurants/hybrid-algorithm/knn/avg_neighbor_knn.pkl",
        description="restaurant knn neighbors profile",
    )
else:
    results_user_knn, results_item_knn = run_new_evaluation(x_range)


plot_comparison_with_labels(results_user_knn, results_item_knn, x_range)
#print(f"Average neighbors count ~ {res_avg}")