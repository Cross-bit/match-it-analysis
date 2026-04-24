#!/bin/python3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
#from clustering.representation import *
from movies_data.initialisation_sampling.clustering.representation import *

# ====================================
# DESCRIPTION
# ====================================
# Evaluation of k-means algorithm with optimal k = 20
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

def find_kmeans_clusters(data, k_clusters: int, cluster_seed: int):
    kmeans = KMeans(n_clusters=k_clusters, random_state=cluster_seed)
    kmeans.fit(data.T)
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

    kmeans = find_kmeans_clusters(data, k_clusters, cluster_seed=np.random.randint(0, 1000))

    # Build a DataFrame similar to fuzzy clusters
    labels = pd.Series(kmeans.labels_, index=movies_representations_scaled.index)

    # Create one-hot cluster distribution DataFrame
    membership_df = pd.get_dummies(labels)
    membership_df = membership_df.astype(float)  # Make sure it's float for consistency
    movies_cluster_distribution_df = membership_df.div(membership_df.sum())

    return sample_movies(number_of_samples, movies_cluster_distribution_df)
