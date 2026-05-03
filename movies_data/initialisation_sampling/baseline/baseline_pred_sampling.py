#!/bin/python3
from surprise import BaselineOnly, Dataset, Reader
import random
from movies_data.initialisation_sampling.clustering.representation import *

# ====================================
# DESCRIPTION
# ====================================
# Uses baseline predictor

popularity_threshold = 40
loader = MovieLensDatasetLoader()
loader.load_data()

movies_df, ratings_df = loader.filter_by_popularity(popularity_threshold)

def prepare_surprise_data(ratings_df):
    # Convert ratings DataFrame to long format (userId, movieId, rating)

    ratings_long = ratings_df.stack().reset_index()
    ratings_long.columns = ['userId', 'movieId', 'rating']

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_long[['userId', 'movieId', 'rating']], reader)
    return data

def train_baseline_model(trainset):
    # Configure BaselineOnly model
    bsl_options = {"method": "sgd", "learning_rate": 0.005, "n_epochs": 20}
    algo = BaselineOnly(bsl_options=bsl_options)

    # Fit model on training data
    algo.fit(trainset)
    return algo

def run_sample_evaluation(number_of_samples = 24, k_clusters = 20, top_k_pool=100):
    # Step 1: Prepare data
    data = prepare_surprise_data(ratings_df)
    trainset = data.build_full_trainset()

    # Step 2: Train Baseline model
    baseline_model = train_baseline_model(trainset)

    # Step 3: Predict baseline scores for all movies
    all_movie_ids = ratings_df.columns
    movie_scores = []

    for movieId in all_movie_ids:
        pred = baseline_model.predict(uid=0, iid=movieId, r_ui=None, verbose=False)
        movie_scores.append((movieId, pred.est))

    # Step 4: Sort movies by predicted rating
    movie_scores_sorted = sorted(movie_scores, key=lambda x: x[1], reverse=True)

    # Step 5: Pick top-k pool (e.g., top 100 movies)
    top_pool = [movieId for movieId, score in movie_scores_sorted[:top_k_pool]]

    # Step 6: Randomly sample top_n movies from top pool
    sampled_movie_ids = set(random.sample(top_pool, number_of_samples))

    return sampled_movie_ids