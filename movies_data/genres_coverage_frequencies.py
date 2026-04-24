#!/bin/python3
import os
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import AXIS_DESC_SIZE, AXIS_VALS_SIZE, IMG_OUTPUT_PATH, HISTOGRAM_EDGECOLOR_1, HISTOGRAM_COLOR_1, TITLE_SIZE

from dataset.data_access import MovieLensDatasetLoader

# ====================================
# DESCRIPTION
# ====================================
# Analysis of movies genre frequency counts.
#

data_loader = MovieLensDatasetLoader()

movies_data_df, ratings_mx_df = data_loader.load_data()


def get_movie_genres_frequencies(movies_data_df: pd.DataFrame):
    genres = movies_data_df.set_index('movieId')['genres'].str.get_dummies("|")
    genres['counts'] = genres.sum(axis=1)

    return genres['counts'].value_counts()


genre_counts_freq = get_movie_genres_frequencies(movies_data_df)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.figure(figsize=(10, 5))
plt.bar(genre_counts_freq.index, genre_counts_freq.values, color=HISTOGRAM_COLOR_1,
edgecolor=HISTOGRAM_EDGECOLOR_1)
plt.xlabel('Počet žánrů', fontsize=AXIS_DESC_SIZE)
plt.ylabel('Počet filmů', fontsize=AXIS_DESC_SIZE)
plt.title('Distribuce počtu žánrů na film', fontsize=TITLE_SIZE)
plt.xticks(genre_counts_freq.index, fontsize=AXIS_VALS_SIZE)
plt.yticks(fontsize=AXIS_VALS_SIZE)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(IMG_OUTPUT_PATH, "genres_distribution.pdf"))
plt.show()

