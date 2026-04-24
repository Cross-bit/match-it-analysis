#!/bin/python3
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from utils.config import IMG_OUTPUT_PATH, AXIS_DESC_SIZE, AXIS_VALS_SIZE, TITLE_SIZE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

# ===================================
# DESCRIPTION
# ===================================
# Creates histogram of ratings matrix density on different datasets
# as we leave out users with specific number of ratings.
#
# Principle: For every cycle (every bar),
# we remove users who have less than k ratings from the matrix.
# We also drop all the columns without any ratings.
# We then measure the density of given matrix.
# density = (# {r_{ij} > 0)/(matrix size)
#
#

#region Different plotting methods...

def plot_histogram_of_results(densities: List[Tuple[float, pd.DataFrame]], max_samples=15):
    """ Plots 2 histograms under each other: places count, users count """

    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    vertical_spacing = 0.4  # mezera mezi grafy (hspace)

    # Extract data
    if max_samples is not None and max_samples > 0:
        col_counts = [s[1].shape[1] for s in densities][:max_samples]
        row_counts = [s[1].shape[0] for s in densities][:max_samples]
    else:
        col_counts = [s[1].shape[1] for s in densities]
        row_counts = [s[1].shape[0] for s in densities]

    x = list(range(1, len(col_counts) + 1))

    # Create 2 vertically stacked subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8),
                            gridspec_kw={'height_ratios': [1, 1]}, dpi=100)

    # === 1. Počet míst ===
    bars_places = axes[0].bar(x, col_counts, color='lightgreen', width=0.5)
    axes[0].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[0].set_ylabel('Počet míst', fontsize=AXIS_DESC_SIZE)
    axes[0].set_title('Vývoj počtu míst s hodnocením', fontsize=TITLE_SIZE)
    axes[0].set_xticks(x)
    axes[0].tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    axes[0].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[0].set_ylim(0, max(col_counts) + 280)

    for bar in bars_places:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.3, str(height),
                    ha='center', va='bottom', fontsize=11)

    # === 2. Počet uživatelů ===
    bars_users = axes[1].bar(x, row_counts, color='salmon', width=0.5)
    axes[1].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[1].set_ylabel('Počet uživatelů (log měřítko)', fontsize=AXIS_DESC_SIZE)
    axes[1].set_title('Vývoj počtu uživatelů v datové sadě', fontsize=TITLE_SIZE)
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    axes[1].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[1].set_ylim(0.9, max(row_counts) * 4)

    for bar in bars_users:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     height * 1.05, str(height),
                    ha='center', va='bottom', fontsize=11)

    plt.subplots_adjust(hspace=vertical_spacing)
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "places-users-coverage.pdf"))
    plt.show()

#endregion

# ==========================
# Evaluation functions
# ==========================

def remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix: pd.DataFrame, min_ratings_count: int) -> pd.DataFrame:
    df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) > min_ratings_count] # drop all users with less than min_ratings_count
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis = 0)] # drop zero columns
    return df_cleaned


def find_densities(rating_matrix, min_it = 0, max_it = 31) -> Tuple[float, pd.DataFrame]:
    """Finds densities for different values of min number of ratings for users in min_it, max_it.
    Returns:
        Tuple[float, pd.DataFrame]: density, cleared matrix
    """

    result = [0 for _ in range(min_it, max_it)]
    for i in range(min_it, max_it):
        cleaned_matrix = remove_users_from_ratings_matrix_by_ratings_count(rating_matrix, i)
        print(i)
        nonzero_count = (cleaned_matrix != 0).sum().sum()
        density = (nonzero_count / cleaned_matrix.size) if cleaned_matrix.size > 0 else 0
        result[i] = (density, cleaned_matrix)
    return result

# ==========================
# Execution
# ==========================

# load original matrix
d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()

densities = find_densities(ratings_matrix)

plot_histogram_of_results(densities)
