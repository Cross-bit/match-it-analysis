import os
import sys
import numpy as np
import pandas as pd

from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx
from dataset.data_access import DatasetLoader

# ===================================
# DESCRIPTION
# ===================================
# Creates basic analysis
#

# load original matrix
d_loader = DatasetLoader()
ratings_matrix: pd.DataFrame = d_loader.load_ratings_matrix()


# Compute stats
nonzero_count = (ratings_matrix != 0).sum().sum()
num_users, num_items = ratings_matrix.shape
total_pairs = ratings_matrix.size
density = (nonzero_count / total_pairs) * 100
avg_ratings_per_user = nonzero_count / num_users
avg_ratings_per_item = nonzero_count / num_items

# Create dataframe for LaTeX table
df = pd.DataFrame({
    "Statistika": [
        "Počet uživatelů",
        "Počet restaurací",
        "Počet hodnocení",
        "Celkový počet dvojic",
        "Hustota",
        "Průměrný počet hodnocení na uživatele",
        "Průměrný počet hodnocení na položku"
    ],
    "Hodnota": [
        f"{num_users}",
        f"{num_items}",
        f"{nonzero_count:,}",
        f"{total_pairs:,}",
        f"{density:.2f} \%",
        f"{avg_ratings_per_user:.2f}",
        f"{avg_ratings_per_item:.2f}"
    ]
}).astype(str)

# Generate LaTeX table
generator = LaTeXTableGeneratorSIUnitx(
    df,
    column_width=1.5
)

latex_code = generator.generate_table(
    caption="Základní statistiky datasetu hodnocení restaurací.",
    label="tab:DatasetStats",
    cell_bold_fn=None
)

print(latex_code)

print("")
#print(f"Total number of values: {number_of_ratings}")
#print(f"Matrix dims: {num_users} x {num_items}")
#print(f"Density: {(density) * 100:.2f} %")
