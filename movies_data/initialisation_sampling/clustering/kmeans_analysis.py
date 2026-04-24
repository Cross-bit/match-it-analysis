#!/bin/python3
import os
from joblib import Parallel, delayed
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#from representation_ import *
from utils.config import AXIS_DESC_SIZE, AXIS_VALS_SIZE, TITLE_SIZE, IMG_OUTPUT_PATH
from movies_data.initialisation_sampling.clustering.representation import get_movies_representation_ml1

# ====================================
# DESCRIPTION
# ====================================
# Detail analysis of movies session initialization sampling using k-means.
# algorithm. Finding the right k.
#

# Create the represenation vectors

movies_representations = get_movies_representation_ml1()

scaler = StandardScaler()
movies_representations_scaled = scaler.fit_transform(movies_representations)
k_range = range(2, 100)

def compute_silhouette_score(k, data):
    print(f"Executing kmeans for {k}")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    return score

silhouette_scores = Parallel(n_jobs=-1)(delayed(compute_silhouette_score)(k, movies_representations_scaled) for k in k_range)

plt.figure(figsize=(11, 7))
plt.plot(k_range, silhouette_scores)

# it oscilates here
filtered_scores = [score for k, score in zip(k_range, silhouette_scores) if k >= 20]
avg_score = sum(filtered_scores) / len(filtered_scores)
plt.axhline(y=avg_score, color='red', linestyle='--', linewidth=1, label=f'Průměr (k ≥ 20): {avg_score:.2f}')  # horizontální čára

plt.xlabel('Počet klastrů', fontsize=AXIS_DESC_SIZE)
plt.ylabel('Silhouette skóre', fontsize=AXIS_DESC_SIZE)
plt.title('Silhouette skóre v závislosti na počtu klastrů', fontsize=TITLE_SIZE)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=AXIS_VALS_SIZE)

plt.xticks(fontsize=AXIS_VALS_SIZE)
plt.yticks(fontsize=AXIS_VALS_SIZE)

plt.savefig(os.path.join(IMG_OUTPUT_PATH, "clustering-silueth-score.pdf"))
plt.show()
#exit()
def compute_inertia(k, data):
    print(f"Executing kmeans for {k}")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    return kmeans.inertia_
k_range = range(1, 100)

wcss = Parallel(n_jobs=-1)(delayed(compute_inertia)(k, movies_representations_scaled) for k in k_range)

kl = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
optimal_k = kl.elbow
print(f"Optimal k: {optimal_k}")

# Plot the elbow chart
plt.figure(figsize=(11, 7))
plt.plot(k_range, wcss, marker='o', markersize=3)
plt.title('Metoda lokte pro určení optimálního k', fontsize=TITLE_SIZE)
plt.xlabel('Počet klastrů (k)', fontsize=AXIS_DESC_SIZE)
plt.ylabel('Součet čtverců vzdáleností uvnitř klastrů (WCSS)', fontsize=AXIS_DESC_SIZE)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.xticks(fontsize=AXIS_VALS_SIZE)
plt.yticks(fontsize=AXIS_VALS_SIZE)

plt.savefig(os.path.join(IMG_OUTPUT_PATH, "clustering-kmeans-elbow-method.pdf"))
plt.show()