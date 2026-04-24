import sys
import os
from matplotlib import pyplot as plt
from utils.config import IMG_OUTPUT_PATH
import numpy as np
from surprise import Dataset, KNNBasic, Reader
from  evaluation_frameworks.general_recommender_evaluation.algorithms.item_knn import ItemItemCFEvaluation
from  evaluation_frameworks.general_recommender_evaluation.algorithms.user_knn import UserKnnCFEvaluation


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()

#region Plotting

def plot_metrics_line_charts(metrics_list):
    """
    Plots line charts for precision@K, recall@K, and ndcg@K metrics.

    Args:
        metrics_list (List[Dict[str, float]]): A list where each dict contains evaluation metrics:
            'precision@K', 'recall@K', and 'ndcg@K'
    """
    # Extract individual metric lists
    precision = [m['precision@K'] for m in metrics_list]
    recall = [m['recall@K'] for m in metrics_list]
    ndcg = [m['ndcg@K'] for m in metrics_list]

    # X values starting at 2
    x = list(range(2, 2 + len(metrics_list)))

    # Create 3 vertically stacked subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=100)

    # Plot precision@K
    axes[0].plot(x, precision, marker='o', linestyle='-', color='blue')
    axes[0].set_title('Precision@K over configurations')
    axes[0].set_ylabel('Precision@K')
    axes[0].set_xticks(x)

    # Plot recall@K
    axes[1].plot(x, recall, marker='o', linestyle='-', color='green')
    axes[1].set_title('Recall@K over configurations')
    axes[1].set_ylabel('Recall@K')
    axes[1].set_xticks(x)

    # Plot ndcg@K
    axes[2].plot(x, ndcg, marker='o', linestyle='-', color='red')
    axes[2].set_title('NDCG@K over configurations')
    axes[2].set_ylabel('NDCG@K')
    axes[2].set_xlabel('Minimum number of user ratings')
    axes[2].set_xticks(x)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(IMG_OUTPUT_PATH, "knn-evaluation-by-dataset-size.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
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

# min number ratings user has to have, otherwise filter out
runs_count = 10


results = []
for min_number_ratings in range(2, 31):
    filtered_dataset = remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, min_number_ratings)
    eval = UserKnnCFEvaluation(filtered_dataset,k=20, algorithm_k=70)
    res = eval.evaluate_crossval(20)
    results.append(res)

plot_metrics_line_charts(results)