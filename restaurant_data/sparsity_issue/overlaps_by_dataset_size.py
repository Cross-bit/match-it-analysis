import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from utils.config import IMG_OUTPUT_PATH
from utils.config import load_from_pickle, save_to_pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_access import *

d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()
ratings_matrix = (ratings_matrix > 0).astype(int)

#region Plotting
def plot_overlap_counts(overlap_counts: list, title: str = "Item-Item Overlap Counts"):
    x = list(range(1, len(overlap_counts) + 1))

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.5), 6), dpi=100)

    ax.plot(x, overlap_counts, marker='o', label='Item-Item Overlaps', color='purple')

    # Dynamic label positioning to avoid floating too far away
    for i, val in enumerate(overlap_counts):
        ax.text(x[i], val + 0.5, f'{val}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Minimal number of user ratings")
    ax.set_ylabel("# of item-item pairs with >1 overlap")
    ax.set_ylim(0, max(overlap_counts) + 3)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "item-overlap-counts.pdf"))
    plt.show()
#endregion

#region Overlap computation
def remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix: pd.DataFrame, min_ratings_count):
    df_cleaned = ratings_matrix[(ratings_matrix != 0).sum(axis=1) >= min_ratings_count]
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    return df_cleaned

def compute_overlap_count(ratings_matrix: pd.DataFrame, threshold: int = 1):
    ratings_sparse = csr_matrix(ratings_matrix.values)
    cooc_matrix = ratings_sparse.T @ ratings_sparse
    cooc_matrix.setdiag(0)
    return (cooc_matrix > threshold).sum()

def run_overlap_analysis():
    min_number_ratings_range = range(1, 32)
    overlap_counts = []

    for i in min_number_ratings_range:
        print(f"Computing overlap for min_ratings >= {i}")
        filtered_matrix = remove_users_from_ratings_matrix_by_ratings_count(ratings_matrix, i)
        count = compute_overlap_count(filtered_matrix)
        overlap_counts.append(count)

    save_to_pickle(
        overlap_counts,
        "restaurants/hybrid-algorithm/sparsity/overlap_counts.pkl",
        description="restaurant overlap counts",
    )

    return overlap_counts
#endregion

# Main
load = False
if load:
    overlap_counts = load_from_pickle(
        "restaurants/hybrid-algorithm/sparsity/overlap_counts.pkl",
        description="restaurant overlap counts",
    )
else:
    overlap_counts = run_overlap_analysis()

plot_overlap_counts(overlap_counts)
