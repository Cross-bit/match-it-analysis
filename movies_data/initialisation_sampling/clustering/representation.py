#!/bin/python3
import pandas as pd
import sys
import os

from dataset.data_access import MovieLensDatasetLoader

# ====================================
# DESCRIPTION
# ====================================
# Representation generation util functions clustering.
#

def get_genres(movies_df, ratings_df):
    # One hot encode genres of each movie
    movies_genres = movies_df.reset_index().set_index('movieId')['genres'].str.get_dummies("|") # (get_dummies handles entire 1 hot encoding)
    # for initialisation we want only movies with a rich description => make sure all the records have at least one genre
    movies_genres = movies_genres[~((movies_genres == 0).all(axis=1)) & movies_genres.index.isin(ratings_df.columns)]
    return movies_genres

# Find popularity
#=================================

def get_popularity(ratings_df):
    popularity_raw = ratings_df.count()
    # Normalize popularity using min max normalisation.
    popularity_normed = (popularity_raw - popularity_raw.min()) / (popularity_raw.max() - popularity_raw.min())
    popularity_normed.name = "popularity"
    return popularity_normed

# Find average movie rating
#=================================
def get_avg_movie_rating(ratings_df):
    average_rating_per_movie = ratings_df.mean().div(5)
    average_rating_per_movie.name = "average_rating"
    return average_rating_per_movie

# Create the represenation vectors
#=================================

def get_movies_representation_ml1(popularity_threshold=100) -> pd.DataFrame:
    """Creates representation of the movies from MovieLens dataset.

    Returns:
        pd.DataFrame: Movie feature vectors by rows. Index: movieId.
    """

    loader = MovieLensDatasetLoader()
    loader.load_data()

    movies_df, ratings_df = loader.filter_by_popularity(popularity_threshold)

    movies_genres = get_genres(movies_df, ratings_df)
    average_rating_per_movie = get_popularity(ratings_df)
    popularity_normed = get_avg_movie_rating(ratings_df)

    movies_representations = pd.concat([movies_genres, average_rating_per_movie, popularity_normed], axis=1)
    return movies_representations



