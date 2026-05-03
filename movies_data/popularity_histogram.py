#!/bin/python3
import matplotlib.pyplot as plt
from dataset.data_access import MovieLensDatasetLoader
from utils.config import AXIS_DESC_SIZE, AXIS_VALS_SIZE, TITLE_SIZE, IMG_OUTPUT_PATH
import os
# ====================================
# DESCRIPTION
# ====================================
# Analysis to find Long-tail distribution
# of the MovieLens dataset.
#

data_loader = MovieLensDatasetLoader()

movies_data_df, ratings_mx_df = data_loader.load_data()

# Calculate movie popularity (number of ratings per movie)
movie_popularity = ratings_mx_df.notna().sum(axis=0)
movie_popularity.sort_values(inplace=True, ascending=False)

samples_count = movie_popularity.size  # Total number of movies

# Define split proportions (adjustable)
head_fraction = 0.10  # Top 10% of movies
middle_fraction = 0.30  # Next 30% of movies
tail_fraction = 0.60  # Bottom 60% of movies (remaining)

# Calculate split indices
head_end = int(samples_count * head_fraction)
middle_end = int(samples_count * (head_fraction + middle_fraction))


plt.figure(figsize=(11, 7), dpi=80)
plt.plot(range(samples_count), movie_popularity, linestyle='-', color='b')  # Line plot with points

# Add vertical lines for head, middle, and tail
plt.axvline(x=head_end, color='r', linestyle='--', label=f'Head část (Top {head_fraction*100}%)')
plt.axvline(x=middle_end, color='g', linestyle='--', label=f'Middle část (Dalších {middle_fraction*100}%)')

plt.xlabel('Movie ID', fontsize=AXIS_DESC_SIZE)
plt.ylabel('Popularita [počet hodnocení]', fontsize=AXIS_DESC_SIZE)
plt.title('Long-tail distribuce popularity filmů', fontsize=TITLE_SIZE)

plt.grid(True)
plt.xticks(
    range(0, samples_count, max(1, samples_count // 10)),
    rotation=45,
    fontsize=AXIS_VALS_SIZE
)
plt.yticks(fontsize=AXIS_VALS_SIZE)
plt.tight_layout()
plt.legend(fontsize=AXIS_VALS_SIZE)
plt.savefig(os.path.join(IMG_OUTPUT_PATH, "long_tail_popularity_movielens.pdf"))
plt.show()

print(f"Total movies: {movie_popularity.size}")
print(f"Maximum ratings for a movie: {movie_popularity.max()}")
