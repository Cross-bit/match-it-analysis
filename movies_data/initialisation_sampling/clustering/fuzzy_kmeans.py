#!/bin/python3
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import skfuzzy as fuzz
#from clustering.representation import *
from movies_data.initialisation_sampling.clustering.representation import *

# ====================================
# DESCRIPTION
# ====================================
# Analysis of movies session initialisation sampling algorithm.
#

# Create the representation vectors

movies_representations = get_movies_representation_ml1()

scaler = StandardScaler()
movies_representations_scaled_np = scaler.fit_transform(movies_representations)


movies_representations_scaled = pd.DataFrame(
    movies_representations_scaled_np,
    index=movies_representations.index,
    columns=movies_representations.columns
)

def find_fuzzy_clusters(data, k_clusters: int, fuzziness: float, cluster_seed: int):
    return fuzz.cluster.cmeans(
        data,               # data.T (features x samples)
        c=k_clusters,       # number of clusters
        error=0.005,        # stopping criterion
        m=fuzziness,        # fuzziness coefficient (usually 2)
        maxiter=1000,       # maximum number of iterations
        init=None,          # initialize randomly
        seed=cluster_seed
    )


def sample_movies(number_of_samples: int, fuzzy_clusters: pd.DataFrame):
    picked_movies = set()
    k_clusters = fuzzy_clusters.shape[1]

    for i in range(number_of_samples):
        c_i = i % k_clusters
        cluster = fuzzy_clusters[c_i]

        # Filter already picked movies
        available = cluster[~cluster.index.isin(picked_movies)]

        # Remove movies with 0 probability
        available = available[available > 0]

        probs = available / available.sum()

        sampled_movie_id = np.random.choice(available.index, p=probs)
        picked_movies.add(sampled_movie_id)

    return picked_movies


def run_sample_evaluation(number_of_samples = 24, k_clusters = 10, fuzziness = 2.0):
    data = movies_representations_scaled.values.T
    cntr, u, u0, d, jm, p, fpc = find_fuzzy_clusters(data, k_clusters, fuzziness, np.random.randint(0, 1000))

    membership_df = pd.DataFrame(u.T, index=movies_representations_scaled.index, columns=[i for i in range(k_clusters)])
    movies_cluster_distribution_df = membership_df.div(membership_df.sum())

    return sample_movies(number_of_samples, movies_cluster_distribution_df)