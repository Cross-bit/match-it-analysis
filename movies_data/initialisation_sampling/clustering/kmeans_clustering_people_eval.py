#!/bin/python3
from joblib import Parallel, delayed
from typing import Set, List, Dict
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pickle
import os
from utils.config import cache_path

from movies_data.initialisation_sampling.clustering.representation import *

popularity_threshold = 40

loader = MovieLensDatasetLoader()
loader.load_data()
movies_df, ratings_df = loader.filter_by_popularity(popularity_threshold)
ratings_zeroed_df = ratings_df.fillna(0)


# region Statistics test functions

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
    metrics = ['popularity', 'coverage', 'norm_entropy']
    methods = list(aggregated_results.keys())

    # Build data for each metric
    data = {metric: [aggregated_results[method][metric] * 100 for method in methods] for metric in metrics}  # 🚀 Rescale to percentages

    x = np.arange(len(methods))  # Label locations
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars for each metric
    for idx, metric in enumerate(metrics):
        bars = ax.bar(x + idx * width, data[metric], width, label=metric.capitalize())

        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',   # 🚀 Round to two decimals
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset label a little above the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Add labels
    ax.set_ylabel('Score (%)')
    ax.set_title('Evaluation Metrics by Method')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig("../../../../img/init-sampling-comparison.pdf")
    plt.show()

# endregion


def custom_correlation_kmeans(data: pd.DataFrame, k_clusters: int, max_iter: int = 100, cluster_seed: int = 42, n_jobs: int = -1):
    np.random.seed(cluster_seed)

    # Randomly initialize centroids
    initial_indices = np.random.choice(len(data), k_clusters, replace=False)
    centroids = data.iloc[initial_indices].values

    def assign_cluster(i, row, centroids):
        similarities = [pearsonr(row, centroid)[0] for centroid in centroids]
        return np.argmax(similarities)

    for iteration in range(max_iter):
        print(f"k: {k_clusters} {iteration}")

        # Step 1: Assign points to clusters (in parallel)
        assignments = Parallel(n_jobs=n_jobs)(
            delayed(assign_cluster)(i, data.iloc[i].values, centroids) for i in range(len(data))
        )

        # Step 2: Recalculate centroids
        new_centroids = []
        for k in range(k_clusters):
            cluster_points = data.iloc[np.where(np.array(assignments) == k)]
            if len(cluster_points) == 0:
                # Reinitialize empty cluster with a random point
                new_centroids.append(data.iloc[np.random.choice(len(data))].values)
            else:
                new_centroids.append(cluster_points.mean().values)

        new_centroids = np.array(new_centroids)

        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    labels_ = pd.Series(assignments, index=data.index)
    return {"labels": labels_, "centroids": centroids}


#
# Sampling functions
#


def sample_movies_from_rating_clusters(
    ratings_df: pd.DataFrame,
    labels: pd.Series,
    num_samples: int = 10,
    top_n_movies: int = 5
) -> Set[int]:
    picked_movies = set()
    k_clusters = labels.nunique()

    for i in range(num_samples):
        c_i = i % k_clusters

        # Pick a random user from cluster
        users_in_cluster = labels[labels == c_i].index
        if len(users_in_cluster) == 0:
            continue

        user_id = np.random.choice(users_in_cluster)
        user_ratings = ratings_df.loc[user_id].dropna()

        if user_ratings.empty:
            continue

        # Get top-rated movies
        top_movies = user_ratings.sort_values(ascending=False).head(top_n_movies)

        # Exclude already picked
        top_movies = top_movies[~top_movies.index.isin(picked_movies)]

        if top_movies.empty:
            continue

        # Sample one
        sampled_movie = np.random.choice(top_movies.index)
        picked_movies.add(sampled_movie)

    return picked_movies
def sample_movies_from_rating_clusters_pool(
    ratings_df: pd.DataFrame,
    labels: pd.Series,
    num_samples: int = 10,
    rating_pool_size: int = 100  # ← pick randomly from top N
) -> Set[int]:
    picked_movies = set()
    k_clusters = labels.nunique()

    for i in range(num_samples):
        c_i = i % k_clusters

        # Pick a random user from cluster
        users_in_cluster = labels[labels == c_i].index
        if len(users_in_cluster) == 0:
            continue

        user_id = np.random.choice(users_in_cluster)
        user_ratings = ratings_df.loc[user_id].dropna()

        if user_ratings.empty:
            continue

        # Randomly pick from top N rated movies
        top_n = user_ratings.sort_values(ascending=False).head(rating_pool_size)
        top_n = top_n[~top_n.index.isin(picked_movies)]

        if top_n.empty:
            continue

        sampled_movie = np.random.choice(top_n.index)
        picked_movies.add(sampled_movie)
    return picked_movies


def evaluate_precision_at_k(ratings_df, recommended_ids, k=5, threshold=3.5):
    """
    ratings_df: DataFrame with users as rows and movieIds as columns (filled with ratings or 0s)
    recommended_ids: list of movieIds that are globally recommended (same for every user)
    k: how many items to evaluate
    threshold: the relevance threshold for a rating
    """
    precisions = []

    for user_id, user_ratings in ratings_df.iterrows():
        # Get the user's relevant items (rated >= threshold)
        relevant_items = set(user_ratings[user_ratings >= threshold].index)

        if not relevant_items:
            continue  # skip users with no relevant items

        # Take top-k from global recommendation list
        top_k = recommended_ids[:k]

        # Count how many are relevant
        hits = sum((int(item) in relevant_items) for item in top_k)

        precisions.append(hits / k)

    return np.mean(precisions)

def run_multiple_sampling_evaluations(
    ratings_df,
    labels,
    movies_df,
    num_runs: int = 10,
    num_samples: int = 24,
    rating_pool_size: int = 100
):
    popularity_scores = []
    coverage_scores = []
    entropy_scores = []

    all_genres = get_unique_genres(movies_df)
    genre_entropy_denom = np.log2(len(all_genres))

    for _ in range(num_runs):
        samples = sample_movies_from_rating_clusters(
            ratings_df,
            labels,
            num_samples=num_samples,
            rating_pool_size=rating_pool_size
        )

        popularity = test_sampled_movies_popularity(samples, ratings_df)
        coverage = test_sampled_movies_coverage(samples, ratings_df)
        normed_entropy = genre_entropy(samples, movies_df) / genre_entropy_denom

        popularity_scores.append(popularity)
        coverage_scores.append(coverage)
        entropy_scores.append(normed_entropy)

    # Average results
    avg_popularity = np.mean(popularity_scores)
    avg_coverage = np.mean(coverage_scores)
    avg_entropy = np.mean(entropy_scores)

    result = {
        "CustomMethod": {
            "popularity": avg_popularity,
            "coverage": avg_coverage,
            "norm_entropy": avg_entropy
        }
    }

    return result


file_name = cache_path("movies", "init-sampling", "kmeans", "stored_people_clustering.pkl")

def run_sample_evaluation(number_of_samples = 24):
    if file_name.exists():
        with open(file_name, 'rb') as f:
            clustered_data = pickle.load(f)
    else:
        clustered_data = custom_correlation_kmeans(ratings_zeroed_df, 17)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(clustered_data, f)

    labels = clustered_data['labels']
    samples = sample_movies_from_rating_clusters(ratings_df, labels, num_samples=number_of_samples)

    return samples