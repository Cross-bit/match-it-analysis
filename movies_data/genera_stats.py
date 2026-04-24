#!/bin/python3

from dataset.data_access import MovieLensDatasetLoader

# ====================================
# DESCRIPTION
# ====================================
# General statistics of movie lens dataset.
#

data_loader = MovieLensDatasetLoader()

_, ratings_mx_df = data_loader.load_data()

avg_number_of_ratings = ratings_mx_df.count().mean()
min_ratings_count = ratings_mx_df.count().min()
max_rating_count = ratings_mx_df.count().max()

density = ratings_mx_df.notna().sum().sum() / ratings_mx_df.size

print(avg_number_of_ratings)
print(min_ratings_count)
print(max_rating_count)
print(f"{density*100:.2f}")

#print(ratings_mx_df.count()[ratings_mx_df.count() > 150])











