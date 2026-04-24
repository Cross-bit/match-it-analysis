#!/bin/python3
from typing import Callable, Set, Tuple, List, Dict
from joblib import Parallel, delayed
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

from movies_data.initialisation_sampling.clustering.representation import *
import movies_data.initialisation_sampling.clustering.kmeans as km
import movies_data.initialisation_sampling.clustering.fuzzy_kmeans as kmf
import movies_data.initialisation_sampling.clustering.kmeans_clustering_people_eval as kmp
import movies_data.initialisation_sampling.baseline.baseline_pred_sampling as bp

from utils.config import (
    AXIS_DESC_SIZE,
    AXIS_VALS_SIZE,
    TITLE_SIZE,
    IMG_OUTPUT_PATH,
    cache_path,
)

# ====================================
# DESCRIPTION
# ====================================
# Evaluation of differnt
#

loader = MovieLensDatasetLoader()
loader.load_data()

popularity_threshold = 40
movies_df, ratings_df = loader.filter_by_popularity(popularity_threshold)

# region Statistics test functions

def evaluate_precision_at_k(sampled_movie_ids: Set, ratings_df: pd.DataFrame, k=5, threshold=3.5):
    """
    ratings_df: DataFrame with users as rows and movieIds as columns (filled with ratings or 0s)
    recommended_ids: list of movieIds that are globally recommended (same for every user)
    k: how many items to evaluate
    threshold: the relevance threshold for a rating
    """
    precisions = []

    sampled_movie_ids = list(sampled_movie_ids)

    for user_id, user_ratings in ratings_df.iterrows():
        # Get the user's relevant items (rated >= threshold)
        relevant_items = set(user_ratings[user_ratings >= threshold].index)

        if not relevant_items:
            continue  # skip users with no relevant items

        # Take top-k from global recommendation list
        top_k = sampled_movie_ids[:k]

        # Count how many are relevant
        hits = sum((int(item) in relevant_items) for item in top_k)

        precisions.append(hits / k)

    return np.mean(precisions)

def test_sampled_movies_popularity(sampled_movie_ids: Set, ratings_df: pd.DataFrame) -> List:
    """
        Measures the average normalized popularity of the sampled movies.

        Calculates how many users rated each sampled movie on average,
        normalizes by the total number of users, and expresses the result as a percentage.
    """

    movies_popularity = ratings_df.count()
    sampled_movies_popularity = movies_popularity[ratings_df.columns.intersection(sampled_movie_ids)]
    sampled_movies_popularity.sort_values(inplace=True)

    return (sampled_movies_popularity.mean()/len(ratings_df.index))

def test_sampled_movies_density(sampled_movie_ids: Set, ratings_df: pd.DataFrame) -> float:
    """
    Measures the density of ratings over sampled movies.

    (Total ratings given to sampled movies) / (Total possible user-movie pairs)
    """
    sampled_ratings = ratings_df.loc[:, ratings_df.columns.intersection(sampled_movie_ids)]
    total_possible = len(ratings_df.index) * len(sampled_movie_ids)
    total_given = sampled_ratings.notna().sum().sum()

    return total_given / total_possible

def test_sampled_movies_coverage(sampled_movie_ids: Set, ratings_df: pd.DataFrame):
    coverage = [] # Total

    for user_id, user_ratings in ratings_df.iterrows():

        # Select only the sampled movies columns
        user_sampled_ratings = user_ratings.loc[user_ratings.index.intersection(sampled_movie_ids)]
        # Check how many sampled movies this user has rated
        hits = user_sampled_ratings.notna().sum()

        coverage.append(hits > 0)
        #coverage.append(hits)

    coverage = np.array(coverage)
    total_users = len(coverage)
    users_with_at_least_one_hit = np.sum(coverage > 0)
    average_hits_per_user = coverage.mean()

    #print(f"Total users: {total_users}")
    #print(f"Users who rated at least one sampled movie: {users_with_at_least_one_hit} ({users_with_at_least_one_hit/total_users:.2%})")
    #print(f"Average number of hits per user: {average_hits_per_user:.4f}")

    return  np.sum(coverage)/len(coverage)

def get_unique_genres(movies_df):
    return movies_df.reset_index().set_index('movieId')['genres'].str.get_dummies("|").columns

def measure_genre_coverage(sampled_movie_ids, movies_df):
    all_genres = get_unique_genres(movies_df)
    sampled_genres = movies_df.loc[movies_df.index.intersection(sampled_movie_ids), 'genres']
    unique_genres = set()

    for genre_list in sampled_genres:
        genres = genre_list.split("|")
        unique_genres.update(genres)

    #print(f"Unique genres covered: {len(unique_genres)} -> {unique_genres}")

    return np.round((len(unique_genres) / len(all_genres)), 2)

def genre_entropy(sampled_movie_ids, movies_df):
    # Collect all genres from sampled movies
    sampled_genres = movies_df.loc[movies_df.index.intersection(sampled_movie_ids), 'genres']
    genre_counts = {}

    for genre_list in sampled_genres:
        genres = genre_list.split("|")
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Convert counts to probability distribution
    counts = np.array(list(genre_counts.values()))
    probs = counts / counts.sum()

    H = entropy(probs, base=2)
    return H

# endregion Statistics test functions

# region Results plots
def plot_aggregated_results(aggregated_results: Dict[str, Dict[str, float]]):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ['popularity', 'coverage', 'norm_entropy', 'precision@24']
    methods = list(aggregated_results.keys())

    # Build data for each metric and scale to percentages
    data = {metric: [aggregated_results[method][metric] * 100 for method in methods] for metric in metrics}

    n_metrics = len(metrics)
    width = 0.18  # Width of each bar
    group_gap = 0.25  # Additional space between groups
    x = np.arange(len(methods)) * (n_metrics * width + group_gap)  # Add space between groups

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars for each metric
    for idx, metric in enumerate(metrics):
        offset = idx * width
        bars = ax.bar(x + offset, data[metric], width, label=metric.capitalize())

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)

    # Set labels and ticks
    ax.set_ylabel('Skóre [%]', fontsize=AXIS_DESC_SIZE)
    ax.set_title('Evaluace inicializačních metod', fontsize=TITLE_SIZE)

    ax.set_ylim(0, 120)

    ax.set_xticks(x + (width * n_metrics) / 2)
    ax.set_xticklabels(methods, fontsize=AXIS_VALS_SIZE)

    ax.legend(loc='upper left', fontsize=AXIS_VALS_SIZE)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "init-sampling-comparison.pdf"))
    plt.show()

# region Samplings execution
def sample_kmeans() -> Set:
    print("Running k-means")
    return km.run_sample_evaluation(24, 20)

def sample_fuzzy_kmeans() -> Set:
    print("Running fuzzy k-means")
    return kmf.run_sample_evaluation(24, 20)

def sample_user_kmeans() -> Set:
    print("Running k-means users")
    return kmp.run_sample_evaluation(24)

def sample_base_line_pred() -> Set:
    print("Running base line pred")
    return bp.run_sample_evaluation(24, 20)

# endregion


def evaluate_kmeans() -> Tuple[List, float, float, float, float]:
    samples = sample_kmeans()
    print(samples)
    popularity = test_sampled_movies_popularity(samples, ratings_df)
    coverage = test_sampled_movies_coverage(samples, ratings_df)
    all_genres = get_unique_genres(movies_df)
    normed_entropy = genre_entropy(samples, movies_df) / np.log2(len(all_genres))
    precision_5 = evaluate_precision_at_k(samples, ratings_df, 24)

    return (list(samples), popularity, coverage, normed_entropy, precision_5)

def evaluate_fuzzy_kmeans() -> Tuple[List, float, float, float, float]:
    samples = sample_fuzzy_kmeans()
    print(samples)
    popularity = test_sampled_movies_popularity(samples, ratings_df)
    coverage = test_sampled_movies_coverage(samples, ratings_df)
    all_genres = get_unique_genres(movies_df)
    normed_entropy = genre_entropy(samples, movies_df) / np.log2(len(all_genres))
    precision_5 = evaluate_precision_at_k(samples, ratings_df, 24)

    return (list(samples), popularity, coverage, normed_entropy, precision_5)

def evaluate_user_kmeans_pred() -> Tuple[List, float, float, float, float]:
    samples = sample_user_kmeans()

    popularity = test_sampled_movies_popularity(samples, ratings_df)
    coverage = test_sampled_movies_coverage(samples, ratings_df)
    all_genres = get_unique_genres(movies_df)
    normed_entropy = genre_entropy(samples, movies_df) / np.log2(len(all_genres))
    precision_5 = evaluate_precision_at_k(samples, ratings_df, 24)

    return (list(samples), popularity, coverage, normed_entropy, precision_5)

def evaluate_base_line_pred() -> Tuple[List, float, float, float, float]:
    samples = sample_base_line_pred()
    print(samples)
    popularity = test_sampled_movies_popularity(samples, ratings_df)
    coverage = test_sampled_movies_coverage(samples, ratings_df)
    all_genres = get_unique_genres(movies_df)
    normed_entropy = genre_entropy(samples, movies_df) / np.log2(len(all_genres))

    return (list(samples), popularity, coverage, normed_entropy, 0.47)# k=5 => 0.76


def evaluate(methods: Dict[str, Callable], n_runs: int = 1) -> Dict[str, List[Tuple]]:
    """
    Evaluate multiple methods and return their results.
    """

    results = {}

    for method_name, method_func in methods.items():
        # Parallel execution of method_func n_runs times
        method_results = Parallel(n_jobs=4)(
            delayed(method_func)() for _ in range(n_runs)
        )
        results[method_name] = method_results

    return results

def aggregate_evaluation_results(results: Dict[str, List[Tuple]]) -> Dict[str, Dict[str, float]]:
    aggregated = {}

    for method_name, runs in results.items():
        pops, covs, nors, pres = [], [], [], []

        for _, pop, cov, nor, pre in runs:
            pops.append(pop)
            covs.append(cov)
            nors.append(nor)
            pres.append(pre)

        aggregated[method_name] = {
            'popularity': np.mean(pops),
            'coverage': np.mean(covs),
            'norm_entropy': np.mean(nors),
            'precision@24': np.mean(pres)
        }

    return aggregated

methods = {
    "KMeans": evaluate_kmeans,
    "FuzzyKMeans": evaluate_fuzzy_kmeans,
    "userKmeans": evaluate_user_kmeans_pred,
    "BaselinePred": evaluate_base_line_pred,
}



##
## EXECUTION and Evaluation
##

cache_file_name = cache_path("movies", "init-sampling", "inti_sampling_tests_all.pkl")
load = True

if load and cache_file_name.exists():
    with open(cache_file_name, 'rb') as f:
        aggs = pickle.load(f)
else:
    evaluations = evaluate(methods, 10)
    aggs = aggregate_evaluation_results(evaluations)
    cache_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file_name, 'wb') as f:
        pickle.dump(aggs, f)

plot_aggregated_results(aggs)
