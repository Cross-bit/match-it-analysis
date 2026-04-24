#!/bin/python3
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from utils.config import HISTOGRAM_COLOR_1, HISTOGRAM_EDGECOLOR_1, IMG_OUTPUT_PATH, AXIS_DESC_SIZE, AXIS_VALS_SIZE, TITLE_SIZE

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

def plot_histogram_of_results3(densities: List[Tuple[float, pd.DataFrame]], max_samples = 15):
    """ Plots 3 histograms under each other: density, left users count, left places count """

    import matplotlib.pyplot as plt
    import pandas as pd

    vertical_spacing = 0.5   # mezera mezi grafy (hspace)

    # Extract data
    if max_samples is not None and max_samples > 0:
        density_percent = [s[0] * 100 for s in densities][:max_samples]
        col_counts = [s[1].shape[1] for s in densities][:max_samples]
        row_counts = [s[1].shape[0] for s in densities][:max_samples]
    else:
        density_percent = [s[0] * 100 for s in densities]
        col_counts = [s[1].shape[1] for s in densities]
        row_counts = [s[1].shape[0] for s in densities]

    x = list(range(1, len(density_percent) + 1))

    # Create 3 vertically stacked subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 12),
                            gridspec_kw={'height_ratios': [1, 1, 1]}, dpi=100)

    # === 1. Hustota matice ===
    bars_density = axes[0].bar(x, density_percent, color='skyblue', width=0.5)
    axes[0].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[0].set_ylabel('Hustota matice (%)', fontsize=AXIS_DESC_SIZE)
    axes[0].set_title('Kumulativní histogram hustoty matice', fontsize=TITLE_SIZE)
    axes[0].set_xticks(x)
    axes[0].tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    axes[0].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[0].set_ylim(0, max(density_percent) + 40)

    for bar in bars_density:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.5, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, rotation=45)

    # === 2. Počet míst ===
    bars_places = axes[1].bar(x, col_counts, color='lightgreen', width=0.5)
    axes[1].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[1].set_ylabel('Počet míst', fontsize=AXIS_DESC_SIZE)
    axes[1].set_title('Kumulativní histogram počtu míst s hodnocením', fontsize=TITLE_SIZE)
    axes[1].set_xticks(x)
    axes[1].tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    axes[1].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[1].set_ylim(0, max(col_counts) + 280)

    for bar in bars_places:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.3, str(height),
                    ha='center', va='bottom', fontsize=11)

    # === 3. Počet uživatelů ===
    bars_users = axes[2].bar(x, row_counts, color='salmon', width=0.5)
    axes[2].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[2].set_ylabel('Počet uživatelů (log měřítko)', fontsize=AXIS_DESC_SIZE)
    axes[2].set_title('Kumulativní histogram počtu uživatelů v datové sadě', fontsize=TITLE_SIZE)
    axes[2].set_yscale('log')
    axes[2].set_xticks(x)
    axes[2].tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    axes[2].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[2].set_ylim(0, max(row_counts) * 4)

    for bar in bars_users:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.3, str(height),
                    ha='center', va='bottom', fontsize=11)

    plt.subplots_adjust(hspace=vertical_spacing)
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "density-histograms.pdf"))
    plt.show()


def plot_histogram_of_results_density_only(densities, celkovy_pocet=-1):
    """
    Vykreslí histogram hustoty matice (v %), kde každý sloupec odpovídá datasetu
    omezenému na uživatele s minimálním počtem hodnocení.
    """

    import matplotlib.pyplot as plt
    import os

    if celkovy_pocet <= 0:
        celkovy_pocet = len(densities)

    if celkovy_pocet > len(densities):
        raise ValueError("Požadovaný počet vzorků přesahuje dostupná data.")

    hustoty = [d[0] * 100 for d in densities[:celkovy_pocet]]
    x = list(range(1, len(hustoty) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, hustoty, color=HISTOGRAM_COLOR_1, edgecolor=HISTOGRAM_EDGECOLOR_1, width=0.5)

    ax.set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    ax.set_ylabel('Hustota (%)', fontsize=AXIS_DESC_SIZE)
    ax.set_title('Vývoj hustoty datasetu', fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    ax.tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    ax.set_ylim(0, max(hustoty) + 40)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom',
                fontsize=10, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "density-only-histogram.pdf"))
    plt.show()

#def plot_histogram_of_results_density_only(densities, celkovy_pocet=-1):
#    """
#    Vykreslí histogram hustoty matice (v %), kde každý sloupec odpovídá datasetu
#    omezenému na uživatele s minimálním počtem hodnocení.
#    """
#
#    import matplotlib.pyplot as plt
#    import os
#
#    if celkovy_pocet <= 0:
#        celkovy_pocet = len(densities)
#
#    if celkovy_pocet > len(densities):
#        raise ValueError("Požadovaný počet vzorků přesahuje dostupná data.")
#
#    hustoty = [d[0] * 100 for d in densities[:celkovy_pocet]]
#    x = list(range(1, len(hustoty) + 1))
#
#    fig, ax = plt.subplots(figsize=(10, 4))
#    bars = ax.bar(x, hustoty, color=HISTOGRAM_COLOR_1, edgecolor=HISTOGRAM_EDGECOLOR_1, width=0.5)
#
#    ax.set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
#    ax.set_ylabel('Hustota (%)', fontsize=AXIS_DESC_SIZE)
#    ax.set_title('Vývoj hustoty datasetu', fontsize=TITLE_SIZE)
#    ax.set_xticks(x)
#    ax.tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
#    ax.tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
#    ax.set_ylim(0, max(hustoty) + 40)
#
#    for bar in bars:
#        height = bar.get_height()
#        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
#                f'{height:.1f}%', ha='center', va='bottom',
#                fontsize=10, rotation=45)
#
#    plt.tight_layout()
#    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "density-and-numbers.pdf"))
#    plt.show()


def plot_histogram_of_results2(densities: List[Tuple[float, pd.DataFrame]]):
    """ Plots 2 histograms under each other: density, left places count"""

    # Extract data
    sparsities_percent = [s[0] * 100 for s in densities]
    col_counts = [s[1].shape[1] for s in densities]

    # X-axis: index positions
    x = list(range(1, len(densities) + 1))

    # Create subplots: 2 rows, shared x-axis
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})

    # === Top subplot: Sparsity bar chart ===
    bars = axes[0].bar(x, sparsities_percent, color='skyblue', width=0.5)

    axes[0].set_xlabel('Minimal number of user ratings')
    axes[0].set_ylabel('Density (%)')
    axes[0].set_title('Cumulative histogram of matrix densities')
    axes[0].set_xticks(x)
    axes[0].set_ylim(0, max(sparsities_percent) + 10)

    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom',
                    fontsize=8, rotation=45)

    # === Bottom subplot: Column count bar chart ===
    col_bars = axes[1].bar(x, col_counts, color='lightgreen', width=0.5)
    axes[1].set_xlabel('Minimal number of user ratings')
    axes[1].set_ylabel('Number of places in dataset')
    axes[1].set_title('Cumulative histogram of number places with rating')
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, max(col_counts) + 5)

    for bar in col_bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.3,
                    str(height),
                    ha='center', va='bottom',
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "density-histograms-user-ratings-only.pdf"))
    plt.show()


def plot_histogram_of_results3_(densities: List[Tuple[float, pd.DataFrame]], max_samples: int = 15):
    """ Plots 3 histograms: density, places count, user count with optional aggregation of tail """
    import matplotlib.pyplot as plt
    import pandas as pd

    from utils.config import AXIS_DESC_SIZE, AXIS_VALS_SIZE, TITLE_SIZE, IMG_OUTPUT_PATH

    # Aggregated data containers
    density_percent = []
    col_counts = []
    row_counts = []

    # Aggregate entries
    for i, (density, df) in enumerate(densities):
        if i < max_samples:
            density_percent.append(density * 100)
            col_counts.append(df.shape[1])
            row_counts.append(df.shape[0])
        elif i == max_samples:
            # Create last "overflow" bin
            density_percent.append(density * 100)
            col_counts.append(df.shape[1])
            row_counts.append(df.shape[0])
        else:
            # Accumulate overflow into last bin
            density_percent[-1] += density * 100
            col_counts[-1] += df.shape[1]
            row_counts[-1] += df.shape[0]

    x = list(range(1, min(max_samples, len(densities)) + 1))
    if len(densities) > max_samples:
        x.append(x[-1] + 1)
        x_labels = [str(i) for i in range(1, max_samples)] + [f'{max_samples}', f'>{max_samples}']
    else:
        x_labels = [str(i) for i in x]

    # Create plots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [1, 1, 1]}, dpi=100)

    # === 1. Matrix density ===
    bars_density = axes[0].bar(x, density_percent, color='skyblue', width=0.5)
    axes[0].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[0].set_ylabel('Hustota matice (%)', fontsize=AXIS_DESC_SIZE)
    axes[0].set_title('Kumulativní histogram hustoty matice', fontsize=TITLE_SIZE)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, fontsize=AXIS_VALS_SIZE)
    axes[0].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[0].set_ylim(0, max(density_percent) + 40)

    for bar in bars_density:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, rotation=45)

    # === 2. Place count ===
    bars_places = axes[1].bar(x, col_counts, color='lightgreen', width=0.5)
    axes[1].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[1].set_ylabel('Počet míst v datové sadě', fontsize=AXIS_DESC_SIZE)
    axes[1].set_title('Kumulativní histogram počtu míst s hodnocením', fontsize=TITLE_SIZE)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, fontsize=AXIS_VALS_SIZE)
    axes[1].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[1].set_ylim(0, max(col_counts) + 280)

    for bar in bars_places:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.3, str(height),
                    ha='center', va='bottom', fontsize=11)

    # === 3. User count ===
    bars_users = axes[2].bar(x, row_counts, color='salmon', width=0.5)
    axes[2].set_xlabel('Minimální počet hodnocení uživatele', fontsize=AXIS_DESC_SIZE)
    axes[2].set_ylabel('Počet uživatelů v datové sadě', fontsize=AXIS_DESC_SIZE)
    axes[2].set_title('Kumulativní histogram počtu uživatelů v datové sadě', fontsize=TITLE_SIZE)
    axes[2].set_yscale('log')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, fontsize=AXIS_VALS_SIZE)
    axes[2].tick_params(axis='y', labelsize=AXIS_VALS_SIZE)
    axes[2].set_ylim(0, max(row_counts) * 4)

    for bar in bars_users:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, height + 0.3, str(height),
                    ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "density-histograms.pdf"))
    plt.show()

def plot_histogram_of_results(sparsities: List[Tuple[float, pd.DataFrame]]):
    """ Plots histograms of matrix density overlayed with lines - left user counts, places count"""
    # Extract data
    sparsities_percent = [s[0] * 100 for s in sparsities]
    row_counts = [len(s[1]) for s in sparsities]
    col_counts = [s[1].shape[1] for s in sparsities]
    non_zero_counts = [df.astype(bool).sum().sum() / (df.shape[0] * df.shape[1]) * 100 for _, df in sparsities]



    # X-axis positions
    x = list(range(1, len(sparsities_percent) + 1))

    # Set up figure and primary axis
    fig, ax1 = plt.subplots(figsize=(max(10, len(x) * 0.5), 6), dpi=100)

    # Bar chart for sparsity
    bars = ax1.bar(x, sparsities_percent, width=0.5, edgecolor='black', label='Sparsity (%)')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_ylim(0, max(sparsities_percent) + 10)
    ax1.set_xticks(x)

    # Label bars with percentage
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=8,
                rotation=45)

    # Secondary y-axis for counts
    ax2 = ax1.twinx()
    ax2.plot(x, row_counts, color='orange', marker='o', label='Row Count')
    ax2.plot(x, col_counts, color='green', marker='s', label='Column Count')
    ax2.plot(x, non_zero_counts, color='blue', marker='^', label='Non-Zero Count')
    ax2.set_ylabel('Row / Column / Non-Zero Count')
    ax2.set_ylim(0, max(max(row_counts), max(col_counts), max(non_zero_counts)) + 10)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Sparsity, Row/Column Counts, and Non-Zero Values')
    plt.tight_layout()
    # plt.savefig('../../../img/user-place-coverage-bar-chart.pdf')
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "density-histograms-lines.pdf"))
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
    print("sdf")
    return result

# ==========================
# Execution
# ==========================

# load original matrix
d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()

densities = find_densities(ratings_matrix)

plot_histogram_of_results_density_only(densities, 15)
# plot_histogram_of_results3(densities)
