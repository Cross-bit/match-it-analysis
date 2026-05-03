#!/bin/python3
from sklearn.discriminant_analysis import StandardScaler
from abc import ABC
from scipy.stats import pearsonr
import numpy as np

from movies_data.initialisation_sampling.clustering.representation import *

# ====================================
# DESCRIPTION
# ====================================
# Evaluation of k-means k=20 with custom algorithm.
#

popularity_threshold = 40

# Create the representation vectors
movies_representations = get_movies_representation_ml1(popularity_threshold)

k_clusters = 20
scaler = StandardScaler()
movies_representations_scaled_np = scaler.fit_transform(movies_representations)

movies_representations_scaled = pd.DataFrame(
    movies_representations_scaled_np,
    index=movies_representations.index,
    columns=movies_representations.columns
)

def custom_correlation_kmeans(data: pd.DataFrame, k_clusters: int, max_iter: int = 100, cluster_seed: int = 42):
    np.random.seed(cluster_seed)

    # Randomly initialize centroids (choose k data points as initial centroids)

    initial_indices = np.random.choice(len(data), k_clusters, replace=False)
    centroids = data.iloc[initial_indices].values

    for iteration in range(max_iter):
        # Step 1: Assign points to clusters based on highest Pearson correlation with centroids
        assignments = []
        for i in range(len(data)):
            similarities = [pearsonr(data.iloc[i], centroid)[0] for centroid in centroids]
            best_cluster = np.argmax(similarities)
            assignments.append(best_cluster)

        # Step 2: Recalculate centroids as the mean of points in each cluster
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


    # Return cluster assignments
    labels_ = pd.Series(assignments, index=data.index)

    return {"labels": labels_, "centroids": centroids}


def find_kmeans_clusters(data, k_clusters: int, cluster_seed: int):
    kmeans = custom_correlation_kmeans(data, k_clusters, cluster_seed=cluster_seed)
    return kmeans

def sample_movies(number_of_samples: int, kmeans_clusters: pd.DataFrame):
    picked_movies = set()
    k_clusters = kmeans_clusters.shape[1]

    for i in range(number_of_samples):
        c_i = i % k_clusters
        cluster = kmeans_clusters[c_i]

        # Filter already picked movies
        available = cluster[~cluster.index.isin(picked_movies)]

        # Uniform probabilities
        probs = np.ones(len(available)) / len(available)

        sampled_movie_id = np.random.choice(available.index, p=probs)
        picked_movies.add(sampled_movie_id)

    return picked_movies

def run_sample_evaluation(number_of_samples = 24, k_clusters = 20):
    data = movies_representations_scaled.values.T

    kmeans = find_kmeans_clusters(movies_representations_scaled, k_clusters, cluster_seed=np.random.randint(0, 1000))
    # Build a DataFrame similar to fuzzy clusters
    labels = pd.Series(kmeans["labels"], index=movies_representations_scaled.index)

    # Create one-hot cluster distribution DataFrame
    membership_df = pd.get_dummies(labels)
    membership_df = membership_df.astype(float)  # Make sure it's float for consistency
    movies_cluster_distribution_df = membership_df.div(membership_df.sum())

    return sample_movies(number_of_samples, movies_cluster_distribution_df)