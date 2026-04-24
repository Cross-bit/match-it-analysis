#!/bin/python3
import pandas as pd
from surprise import Dataset, Reader, BaselineOnly
from surprise.model_selection import train_test_split
from collections import defaultdict
from joblib import Parallel, delayed
from collections import defaultdict
from typing import Set, Tuple, List, Dict
import numpy as np
from movies_data.initialisation_sampling.clustering.representation import *

popularity_threshold = 40
movies_df, ratings_df = filter_by_popularity(load_ml1(), popularity_threshold)
ratings_zeroed_df = ratings_df.fillna(0)

# --- Convert to Surprise format ---
# Melt DataFrame to long format: userId, movieId, rating
long_df = ratings_zeroed_df.reset_index().melt(id_vars='userId', var_name='movieId', value_name='rating')

# Filter out zeros (unrated)
long_df = long_df[long_df['rating'] > 0]

# Define Surprise reader
reader = Reader(rating_scale=(0.5, 5.0))  # Assuming ML ratings

# Load dataset
data = Dataset.load_from_df(long_df[['userId', 'movieId', 'rating']], reader)

# --- Train BaselineOnly ---
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
algo = BaselineOnly()
algo.fit(trainset)
predictions = algo.test(testset)

# --- Build top-N recommendations per user ---
def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for (iid, _) in user_ratings[:n]]
    return top_n

# --- Evaluate precision@k ---
def precision_at_k(top_n, testset, k=5, threshold=3.5):
    relevant = defaultdict(set)
    for uid, iid, true_r in testset:
        if true_r >= threshold:
            relevant[uid].add(iid)

    precisions = []
    for uid in top_n:
        recommended = top_n[uid][:k]
        if not relevant[uid]:
            continue
        hits = sum((iid in relevant[uid]) for iid in recommended)
        precisions.append(hits / k)
    return sum(precisions) / len(precisions)

# --- Run evaluation ---
top_n = get_top_n(predictions, n=24)
precision = precision_at_k(top_n, testset, k=24)

print(f"BaselineOnly Precision@24: {precision:.4f}")