#!/bin/python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.data_access import MovieLensDatasetLoader
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserSparse

from utils.config import (
    AXIS_DESC_SIZE,
    AXIS_VALS_SIZE,
    IMG_OUTPUT_PATH,
    HISTOGRAM_EDGECOLOR_1,
    HISTOGRAM_COLOR_1,
    TITLE_SIZE
)

# ====================================
# CONFIG
# ====================================

TOP_K = 10
SAMPLE_USERS = 2000
TOP_ITEMS = 8000

USE_RECENCY_RERANK = True

# options: "multiplicative" | "linear"
RERANK_MODE = "multiplicative"

RECENCY_PARAM = 0.8   # beta (multiplicative) / alpha (linear)
RECENCY_LAMBDA = 0.1

CANDIDATE_POOL = 2000


# ====================================
# RECENCY RERANK FUNCTION
# ====================================

def recency_rerank(
    scores,
    candidate_indices,
    movie_id_map,
    movie_year_map,
    param,
    lambda_,
    current_year,
    mode
):

    rescored = []

    candidate_scores = scores[candidate_indices]

    if mode == "linear":
        norm_scores = (candidate_scores - candidate_scores.min()) / (
            candidate_scores.max() - candidate_scores.min() + 1e-8
        )

    for i, idx in enumerate(candidate_indices):

        movie_id = movie_id_map[idx]
        year = movie_year_map.get(movie_id, current_year)

        recency = np.exp(-lambda_ * (current_year - year))

        if mode == "multiplicative":

            beta = param
            new_score = scores[idx] * (beta + (1 - beta) * recency)

        elif mode == "linear":

            alpha = param
            norm_score = norm_scores[i]
            new_score = (1 - alpha) * norm_score + alpha * recency

        else:
            raise ValueError("Unknown rerank mode")

        rescored.append((idx, new_score))

    rescored.sort(key=lambda x: x[1], reverse=True)

    return [idx for idx, _ in rescored]


# ====================================
# LOAD DATA
# ====================================

print("Loading dataset...")

loader = MovieLensDatasetLoader("ml-32m")
movies_df, ratings_csr, user_id_map, movie_id_map = loader.load_sparse_ratings()


# ====================================
# FILTER POPULAR ITEMS
# ====================================

print("Filtering most popular items...")

item_popularity = np.array((ratings_csr > 0).sum(axis=0)).flatten()

top_items = np.argsort(item_popularity)[-TOP_ITEMS:]

ratings_csr = ratings_csr[:, top_items]

movie_id_map = {
    new_idx: movie_id_map[old_idx]
    for new_idx, old_idx in enumerate(top_items)
}

print("Items after filtering:", ratings_csr.shape[1])


# ====================================
# EXTRACT YEARS
# ====================================

movies_df["year"] = (
    movies_df["title"]
    .str.extract(r"\((\d{4})\)", expand=False)
    .astype(float)
)

movie_year_map = dict(zip(movies_df["movieId"], movies_df["year"]))

current_year = int(movies_df["year"].max())


# ====================================
# TRAIN EASE
# ====================================

print("Training EASE...")

easer = EaserSparse(l2=5000)
easer.fit(ratings_csr, user_id_map, movie_id_map)


# ====================================
# COLLECT DATA
# ====================================

print("Collecting recommendations...")

dataset_years = [
    y for y in movie_year_map.values()
    if not pd.isna(y)
]

ease_years = []
reranked_years = []

users = list(user_id_map.values())[:SAMPLE_USERS]


for user_id in tqdm(users):

    profile = easer.get_user_vector(user_id)

    scores = easer.get_item_scores(profile)

    known = np.where(profile > 0)[0]
    scores[known] = -np.inf


    # ---------- EASE TOP K ----------

    top_k = np.argsort(scores)[-TOP_K:][::-1]

    for idx in top_k:

        movie_id = movie_id_map[idx]
        year = movie_year_map.get(movie_id)

        if not pd.isna(year):
            ease_years.append(year)


    # ---------- RECENCY RERANK ----------

    if USE_RECENCY_RERANK:

        candidate_pool = np.argsort(scores)[-CANDIDATE_POOL:]

        reranked = recency_rerank(
            scores,
            candidate_pool,
            movie_id_map,
            movie_year_map,
            RECENCY_PARAM,
            RECENCY_LAMBDA,
            current_year,
            RERANK_MODE
        )[:TOP_K]

        for idx in reranked:

            movie_id = movie_id_map[idx]
            year = movie_year_map.get(movie_id)

            if not pd.isna(year):
                reranked_years.append(year)


# ====================================
# PLOT
# ====================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]

plt.figure(figsize=(14, 6))

bins = np.arange(1920, 2025, 5)

plt.hist(
    dataset_years,
    bins=bins,
    alpha=0.4,
    label="Dataset",
    color=HISTOGRAM_COLOR_1,
    edgecolor=HISTOGRAM_EDGECOLOR_1
)

plt.hist(
    ease_years,
    bins=bins,
    alpha=0.7,
    label="Doporučení (EASE)"
)

if USE_RECENCY_RERANK:

    if RERANK_MODE == "multiplicative":
        label = f"EASE + Recency (multiplicative β={RECENCY_PARAM})"
    else:
        label = f"EASE + Recency (linear α={RECENCY_PARAM})"

    plt.hist(
        reranked_years,
        bins=bins,
        alpha=0.7,
        label=label
    )


plt.xlabel("Rok vydání filmu", fontsize=AXIS_DESC_SIZE)
plt.ylabel("Počet filmů", fontsize=AXIS_DESC_SIZE)

plt.title(
    "Distribuce roku vydání filmů v datasetu a v doporučeních",
    fontsize=TITLE_SIZE
)

plt.xticks(fontsize=AXIS_VALS_SIZE, rotation=45)
plt.yticks(fontsize=AXIS_VALS_SIZE)

plt.legend()

plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()


# ====================================
# SAVE FILE
# ====================================

if RERANK_MODE == "multiplicative":
    filename = f"recommendation_year_distribution_ease_multiplicative_beta_{RECENCY_PARAM}.pdf"
else:
    filename = f"recommendation_year_distribution_ease_linear_alpha_{RECENCY_PARAM}.pdf"

plt.savefig(
    os.path.join(
        IMG_OUTPUT_PATH,
        filename
    )
)

plt.show()