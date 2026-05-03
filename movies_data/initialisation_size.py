#!/bin/python3
import os
from joblib import Parallel, delayed
from typing import Tuple, List
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import copy

from dataset.data_access import MovieLensDatasetLoader
from utils.config import (
    AXIS_DESC_SIZE,
    AXIS_VALS_SIZE,
    TITLE_SIZE,
    IMG_OUTPUT_PATH,
    load_from_pickle,
    save_to_pickle,
)


# ====================================
# DESCRIPTION
# ====================================
# Analysis to find optimal size of initialisation
# session.
#

data_loader = MovieLensDatasetLoader()

opp, ratings_mx_df = data_loader.load_data() #load_ml1()
opp, ratings_mx_df_pop = data_loader.filter_by_popularity(40) #load_ml1()

ratings_mx_df.fillna(0, inplace=True) # Algorithm relies on this, nan is infeasible
ratings_mx_df_pop.fillna(0, inplace=True) # Algorithm relies on this, nan is infeasible


user_counts = ratings_mx_df.reset_index().set_index("userId").count(axis=1)


# region Prediction and recommendation

def predict_rating_for_item_knn(ratings_mx_df, target_user, target_item, neighbors_df):
    """ Implementation of typical prediction function for knn. """

    sims = neighbors_df.set_index("neighbors")["similarities"]


    # Filter out neighbors who haven't rated the target item
    valid_neighbors = sims[~ratings_mx_df.loc[sims.index, target_item].eq(0)]

    if valid_neighbors.empty:
        return ratings_mx_df.loc[target_user].mean()  # fallback

    neighbor_ratings = ratings_mx_df.loc[valid_neighbors.index, target_item]
    neighbor_means = ratings_mx_df.loc[valid_neighbors.index].mean(axis=1)

    numer = ((neighbor_ratings - neighbor_means) * valid_neighbors).sum()
    denom = valid_neighbors.abs().sum()

    user_mean = ratings_mx_df.loc[target_user].mean()
    return user_mean + (numer / denom) if denom != 0 else user_mean

def find_all_predictions_for_user(ratings_mx_df: pd.DataFrame, user_id: int, neighbors_df: pd.DataFrame):
    """ Find neighbor items with top k predictions. """

    predictions = []

    for movie_id in ratings_mx_df.loc[user_id].index:
        predictions.append(predict_rating_for_item_knn(ratings_mx_df, user_id, movie_id, neighbors_df))

    predictions_df = pd.DataFrame([predictions], columns=ratings_mx_df.columns, index=[user_id])


    #print(predictions_df)
    #print(predictions_df.sort_values(axis=1, by=user_id, ascending=False))

    return predictions_df.sort_values(axis=1, by=user_id, ascending=False)


def get_all_predictions_for_test_items(ratings_mx_df: pd.DataFrame, user_id: int, neighbors_df: pd.DataFrame, test_items: pd.DataFrame):
    predictions = find_all_predictions_for_user(ratings_mx_df, user_id, neighbors_df)
    return predictions[test_items]

def find_neighbors(mx_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    knn = NearestNeighbors(metric='correlation', algorithm='brute')
    knn_model = knn.fit(mx_df)

    group_ratings_knn_query = mx_df.loc[user_id].values.reshape(1, -1)

    similarities, indices = knn_model.kneighbors(group_ratings_knn_query, n_neighbors=10)

    neighbors_indices = indices[0][1:]
    similar_users = mx_df.index[neighbors_indices]
    similarities = similarities[0][1:]

    return pd.DataFrame(data={ "neighbors": similar_users, "similarities": similarities })

#endregion

# region evaluation functions
def rmse(mx: pd.DataFrame, user_id: int, predicted_ratings: pd.DataFrame, train_movies: List, test_movies: List):
    """ Finds RMSE between for  """

    mx = mx.drop(columns=train_movies)

    user_ratings = mx.loc[user_id, test_movies]
    predicted_ratings = predicted_ratings.loc[user_id, test_movies]

    nom = sum((user_ratings - predicted_ratings)**2)
    denom = user_ratings.size

    if (denom == 0):
        print("Warning rmse sample size was zero!")
        return 0

    return np.sqrt(nom/denom)

def count_number_of_positives(mx_df, similar_users, test_movies):
    # count number of positives
    recommendation_candidates = mx_df.loc[similar_users.neighbors].apply(lambda row: row[row != 0].index.to_list(), axis=1)
    recommendation_candidates_unique = set([num for sublist in recommendation_candidates.values for num in sublist])
    common_candidates = set(test_movies.to_list()).intersection(recommendation_candidates_unique)

    return len(common_candidates)
# endregion

# region evaluation
def get_training_and_test_movies(mx_df: pd.DataFrame, user_id: int, ratings_count: int):
    user_ratings: pd.Series = mx_df.loc[user_id]

    # Randomly select k ratings as train-user's sample
    train_ratings = user_ratings[~user_ratings.eq(0)].sample(n=ratings_count, random_state=np.random.randint(1000))
    train_movies = train_ratings.index

    # All the ratings user did not rated (=> we want to recommend these)
    test_ratings = user_ratings[(~user_ratings.index.isin(train_movies))]
    test_movies = test_ratings[~test_ratings.eq(0)].index #all the movies we expect to be recommended
    return (train_movies, test_movies)

def proceed_evaluation_rmse(ratings_mx_df: pd.DataFrame, user_id: int, k: int):

    mx_df =  ratings_mx_df.copy()

    # Get test-user's training data
    user_ratings: pd.Series = mx_df.loc[user_id]
    train_movies, test_movies = get_training_and_test_movies(ratings_mx_df, user_id, k)

    # remove all the test ratings from the ratings matrix (user rated only the k-th selected train data)
    mx_df.loc[user_id, test_movies.values] = 0

    similar_users = find_neighbors(mx_df, user_id)
    predictions = find_all_predictions_for_user(ratings_mx_df, user_id, similar_users)

    predictions = predictions[test_movies]
    return rmse(ratings_mx_df, user_id, predictions, train_movies, test_movies)

# endregion evaluation
def find_accuracies(ratings_mx_df, k, min_ratings_threshold, users_sample_size) ->List[float]:
    """ Evaluates accuracy using RMSE for random users with k ratings.
    Args:
        ratings_mx_df (pd.DataFrame): Expects user ratings
        k (int): number of user ratings - the number of ratings "we pretend" user has.
        min_ratings_threshold: Minimal number of ratings user has to have to be chosen into the calculation.
        users_sample_size: Number of users to sample.
    Returns:
        float: RME value.
    """

    mx = ratings_mx_df.reset_index().set_index("userId")

    # Get required number of users with enough ratings
    suitable_users: pd.DataFrame = mx[(mx != 0).sum(axis=1) > min_ratings_threshold]


    if (len(suitable_users) < users_sample_size):
        print("Not enough suitable users in the dataset; Adjust the [users_sample_size] parameter.")
        exit(1)

    np.random.seed(42) # make sure to always choose the same random users
    sampled_users = np.random.choice(suitable_users.index, size=users_sample_size, replace=False)

    sampled_users_acc = []
    for user_id in sampled_users:
        proceed_evaluation_rmse(ratings_mx_df, user_id, k)
        sampled_users_acc.append(proceed_evaluation_rmse(ratings_mx_df, user_id, k))

    return sampled_users_acc

# Define your function to find accuracies
def find_accuracies_wrapper(ratings_mx_df, k, min_rating_threshold, sample_size):
    print(f"Starting task for k={k}, min_rating_threshold={min_rating_threshold}, sample_size={sample_size}")

    accuracy_measures = find_accuracies(ratings_mx_df, k, min_rating_threshold, sample_size)
    print(f"Finished task for range k={k}, min_rating_threshold={min_rating_threshold} with result: {accuracy_measures}")
    return accuracy_measures  # Return numeric result instead of string


def run_sampling(ratings_mx_df, file_name, load = False) -> Tuple[List, List]:

    # Define the range of parameters
    param_range = list(range(5, 100, 5))
    cache_key = f"movies/init-sampling/{file_name}"

    if (load):
        return load_from_pickle(cache_key, description=file_name)

    # Execute tasks with parallelism and print statements
    all_users_samples = Parallel(n_jobs=4)(
        delayed(find_accuracies_wrapper)(ratings_mx_df, i, i + 100, 10) for i in param_range
    )

    # Calculate the mean for each result
    rmse_stat = [np.mean(sample) for sample in all_users_samples]

    save_to_pickle((param_range, rmse_stat), cache_key, description=file_name)

    return (param_range, rmse_stat)


def add_plot(sub_plot, rmse_stat: List, param_range, window_size = 10):

    l_smooth = np.convolve(rmse_stat, np.ones(window_size)/window_size, mode='valid')

    # Create matching param_range for smoothed values
    param_range_smooth = param_range[(window_size - 1)//2 : -(window_size // 2)] if window_size > 1 else param_range

    # First plot (Original)
    sub_plot.plot(param_range, rmse_stat, linestyle='-', color='blue', label='Original')
    sub_plot.plot(param_range_smooth, l_smooth, linestyle='--', color='red', alpha=0.6, linewidth=2, label='Trend (Moving Avg)')
    sub_plot.set_xlabel('Movie ID')
    sub_plot.set_ylabel('Popularity [number of ratings]')
    sub_plot.set_title('Long-tail distribution of movie popularity (Plot 1)')
    sub_plot.grid(True)
    sub_plot.legend()



param_range, rmse_stat = run_sampling(ratings_mx_df, "results_single2.pkl", load=True)
param_range_pop, rmse_stat_pop = run_sampling(ratings_mx_df_pop, "results_popularity2.pkl", load=True)

plt.figure(figsize=(10, 6), dpi=100)

plt.plot(param_range, rmse_stat, linestyle='-', color='blue', label='Všechny filmy')
plt.plot(param_range_pop, rmse_stat_pop, linestyle='-', color='red', label='Pouze populární filmy (Top 40 %)')

window_size = 5
l_smooth = np.convolve(rmse_stat, np.ones(window_size)/window_size, mode='valid')
param_range_smooth = param_range[(window_size - 1)//2 : -(window_size // 2)] if window_size > 1 else param_range

l_smooth_pop = np.convolve(rmse_stat_pop, np.ones(window_size)/window_size, mode='valid')
param_range_smooth_pop = param_range_pop[(window_size - 1)//2 : -(window_size // 2)] if window_size > 1 else param_range_pop

plt.plot(param_range_smooth, l_smooth, linestyle='--', color='blue', alpha=0.35, linewidth=1, label='Trend (klouzavý průměr) – všechny')
plt.plot(param_range_smooth_pop, l_smooth_pop, linestyle='--', color='red', alpha=0.35, linewidth=1, label='Trend (klouzavý průměr) – populární')

plt.xlabel('Počet hlasů uživatelů (k)', fontsize=AXIS_DESC_SIZE)
plt.ylabel('RMSE', fontsize=AXIS_DESC_SIZE)
plt.title('Míra RMSE pro různé hodnoty k', fontsize=TITLE_SIZE)

plt.xticks(fontsize=AXIS_VALS_SIZE)
plt.yticks(fontsize=AXIS_VALS_SIZE)
plt.legend(fontsize=AXIS_VALS_SIZE)

plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_OUTPUT_PATH, "init_sample_size_accuracy2_popularity_comp.pdf"))
plt.show()
