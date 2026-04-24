#!/bin/python3
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils.config import (
    AXIS_DESC_SIZE,
    AXIS_VALS_SIZE,
    IMG_OUTPUT_PATH,
    HISTOGRAM_EDGECOLOR_1,
    HISTOGRAM_COLOR_1,
    TITLE_SIZE
)

from dataset.data_access import MovieLensDatasetLoader


# ====================================
# DESCRIPTION
# ====================================
# Total number of ratings for movies
# released in each year
# ====================================

data_loader = MovieLensDatasetLoader()

print("Loading dataset...")
movies_df, ratings_mx_df = data_loader.load_data()


def get_total_ratings_by_release_year(
    movies_df: pd.DataFrame,
    ratings_mx_df: pd.DataFrame
) -> pd.Series:

    steps = tqdm(total=5, desc="Processing")

    # remove accidental header row
    movies = movies_df[movies_df["movieId"] != "movieId"].copy()
    steps.update(1)

    # convert movieId
    movies["movieId"] = pd.to_numeric(movies["movieId"])
    steps.update(1)

    # extract year
    movies["year"] = (
        movies["title"]
        .str.extract(r"\((\d{4})\)", expand=False)
        .astype(float)
    )
    steps.update(1)

    # count ratings per movie
    ratings_per_movie = ratings_mx_df.count(axis=0)
    ratings_per_movie.index = pd.to_numeric(ratings_per_movie.index)
    steps.update(1)

    # join movieId -> year
    df = pd.DataFrame({
        "movieId": ratings_per_movie.index,
        "ratings": ratings_per_movie.values
    })

    df = df.merge(movies[["movieId", "year"]], on="movieId", how="left")

    ratings_by_year = (
        df.dropna(subset=["year"])
        .groupby("year")["ratings"]
        .sum()
        .sort_index()
    )

    steps.update(1)
    steps.close()

    return ratings_by_year


ratings_by_year = get_total_ratings_by_release_year(movies_df, ratings_mx_df)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]

plt.figure(figsize=(14, 6))

plt.bar(
    ratings_by_year.index,
    ratings_by_year.values,
    color=HISTOGRAM_COLOR_1,
    edgecolor=HISTOGRAM_EDGECOLOR_1
)

plt.xlabel("Rok vydání filmu", fontsize=AXIS_DESC_SIZE)
plt.ylabel("Celkový počet hodnocení", fontsize=AXIS_DESC_SIZE)
plt.title("Popularita filmů podle roku vydání", fontsize=TITLE_SIZE)

plt.xticks(fontsize=AXIS_VALS_SIZE, rotation=45)
plt.yticks(fontsize=AXIS_VALS_SIZE)

plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(IMG_OUTPUT_PATH, "movie_popularity_by_release_year.pdf"))
plt.show()
