#!/usr/bin/env python3

import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parents[1]

def load_allowed_movie_ids_from_links(path: Path) -> set[int]:
    """
    Load allowed MovieLens movieIds from repaired links CSV.
    Expected column: movieId
    """

    allowed = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "movieId" not in reader.fieldnames:
            raise ValueError("links file must contain movieId column")
        for row in reader:
            mid = row.get("movieId")
            if mid and mid.isdigit():
                allowed.add(int(mid))
    return allowed


def extract_year_from_title(title: str) -> float:
    """Extract year from movie title, e.g., 'Toy Story (1995)' -> 1995.0"""
    match = re.search(r'\((\d{4})\)', title)
    return float(match.group(1)) if match else np.nan


def compute_weighted_popularity(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    recency_weight: float = 0.3,
    current_year: int = 2025,
) -> pd.Series:
    """
    Compute popularity score weighted by recency.

    Score = rating_count * (1 + recency_weight * normalized_age)

    Where normalized_age is 0 for oldest movies, 1 for newest.
    """
    # Count ratings per movie
    movie_popularity = ratings.groupby("movieId").size()

    # Extract years
    movies_with_year = movies.copy()
    movies_with_year["year"] = movies_with_year["title"].apply(extract_year_from_title)

    # Merge popularity with year info
    pop_df = pd.DataFrame({
        "movieId": movie_popularity.index,
        "rating_count": movie_popularity.values
    })
    pop_df = pop_df.merge(movies_with_year[["movieId", "year"]], on="movieId", how="left")

    # Handle missing years (use median or set low priority)
    pop_df["year"].fillna(pop_df["year"].median(), inplace=True)

    # Normalize year to [0, 1] range (newer = higher)
    min_year = pop_df["year"].min()
    max_year = pop_df["year"].max()
    pop_df["year_normalized"] = (pop_df["year"] - min_year) / (max_year - min_year)

    # Compute weighted score
    pop_df["weighted_score"] = pop_df["rating_count"] * (1 + recency_weight * pop_df["year_normalized"])

    return pop_df.set_index("movieId")["weighted_score"]


def iterative_filter(
    ratings: pd.DataFrame,
    min_user_interactions: int,
    min_movie_interactions: int,
    max_iter: int = 5,
) -> pd.DataFrame:
    """
    Removes users with small number of interactions and non-popular movies.
    """
    for i in range(max_iter):
        n_before = len(ratings)

        # filter users
        user_counts = ratings.groupby("userId").size()
        active_users = user_counts[user_counts >= min_user_interactions].index
        ratings = ratings[ratings["userId"].isin(active_users)]

        # filter movies
        movie_counts = ratings.groupby("movieId").size()
        popular_movies = movie_counts[movie_counts >= min_movie_interactions].index
        ratings = ratings[ratings["movieId"].isin(popular_movies)]

        n_after = len(ratings)
        print(f"🔁 Iter {i+1}: {n_before:,} → {n_after:,} ratings")

        if n_before == n_after:
            break

    return ratings


def sample_ratings_stratified(
    ratings: pd.DataFrame,
    max_ratings: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample ratings while preserving user and movie distributions.
    """
    if len(ratings) <= max_ratings:
        return ratings

    print(f"📉 Sampling {max_ratings:,} ratings from {len(ratings):,}...")

    # Sample fraction
    frac = max_ratings / len(ratings)

    # Stratified sampling by user (preserves user activity distribution)
    sampled = ratings.groupby("userId", group_keys=False).apply(
        lambda x: x.sample(frac=min(frac * 1.2, 1.0), random_state=seed)
    ).sample(n=min(max_ratings, len(ratings)), random_state=seed)

    print(f"   Sampled: {len(sampled):,} ratings")
    print(f"   Users:   {sampled['userId'].nunique():,}")
    print(f"   Movies:  {sampled['movieId'].nunique():,}")

    return sampled


def main(args):
    input_dir = Path(args.input_dir)
    print(input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading data...")

    ratings = pd.read_csv(
        input_dir / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": int, "movieId": int, "rating": float},
    )

    movies = pd.read_csv(
        input_dir / "movies.csv",
        dtype={"movieId": int, "title": str, "genres": str},
    )

    if args.links_filter:

        links_file_path = Path(input_dir / args.links_filter)
        allowed_movie_ids = load_allowed_movie_ids_from_links(links_file_path)

        before_ratings = len(ratings)
        before_movies = len(movies)

        ratings = ratings[ratings["movieId"].isin(allowed_movie_ids)]
        movies = movies[movies["movieId"].isin(allowed_movie_ids)]

        print("\n🔒 Applied links-based movieId filter")
        print(f"   Allowed movies: {len(allowed_movie_ids):,}")
        print(f"   Ratings: {before_ratings:,} → {len(ratings):,}")
        print(f"   Movies:  {before_movies:,} → {len(movies):,}")

    print(f"📊 Original dataset:")
    print(f"   Ratings: {len(ratings):,}")
    print(f"   Movies:  {len(movies):,}")
    print(f"   Users:   {ratings['userId'].nunique():,}")

    # --------------------------------------------------
    # Sample ratings if needed
    # --------------------------------------------------
    if args.max_ratings and args.max_ratings > 0:
        ratings = sample_ratings_stratified(ratings, args.max_ratings, seed=args.seed)

    # --------------------------------------------------
    # Top N most popular movies (WEIGHTED BY RECENCY)
    # --------------------------------------------------
    print(f"\n🎯 Selecting top {args.top_n_movies} movies (recency_weight={args.recency_weight})...")

    weighted_popularity = compute_weighted_popularity(
        ratings,
        movies,
        recency_weight=args.recency_weight
    )

    top_movie_ids = (
        weighted_popularity
        .sort_values(ascending=False)
        .head(args.top_n_movies)
        .index
    )

    ratings = ratings[ratings["movieId"].isin(top_movie_ids)]

    print(f"   After top-N filter: {len(ratings):,} ratings")

    # --------------------------------------------------
    # Iterative filter
    # --------------------------------------------------
    print("\n🔄 Running iterative filtering...")

    ratings = iterative_filter(
        ratings,
        min_user_interactions=args.min_user_interactions,
        min_movie_interactions=args.min_movie_interactions,
        max_iter=args.max_iter,
    )

    # --------------------------------------------------
    # Filter movies dataset
    # --------------------------------------------------
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]

    # Add year column for reference
    movies["year"] = movies["title"].apply(extract_year_from_title)

    # --------------------------------------------------
    # Stats
    # --------------------------------------------------
    print("\n📊 FINAL DATASET")
    print(f"   Users:   {ratings['userId'].nunique():,}")
    print(f"   Movies:  {ratings['movieId'].nunique():,}")
    print(f"   Ratings: {len(ratings):,}")
    print(f"   Density: {len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()):.4f}")

    # Year distribution
    year_stats = movies["year"].describe()
    print(f"\n📅 Year distribution:")
    print(f"   Median:  {year_stats['50%']:.0f}")
    print(f"   Mean:    {year_stats['mean']:.1f}")
    print(f"   Range:   {year_stats['min']:.0f} - {year_stats['max']:.0f}")

    # --------------------------------------------------
    # Year distribution (detailed)
    # --------------------------------------------------

    years = movies["year"].dropna().astype(int)
    total = len(years)

    year_stats = years.describe()

    print(f"\n📅 Year distribution (movies with known year: {total:,})")
    print(f"   Min:     {int(year_stats['min'])}")
    print(f"   Max:     {int(year_stats['max'])}")
    print(f"   Median:  {int(year_stats['50%'])}")
    print(f"   Mean:    {year_stats['mean']:.1f}")

    # ---- Specific cutoffs ----
    def pct(cond):
        return 100 * cond.sum() / total if total else 0

    print("\n🎯 Year cutoffs:")
    print(f"   ≥ 2000: {pct(years >= 2000):5.1f}%")
    print(f"   ≥ 2010: {pct(years >= 2010):5.1f}%")
    print(f"   ≥ 2015: {pct(years >= 2015):5.1f}%")
    print(f"   ≥ 2020: {pct(years >= 2020):5.1f}%")

    # ---- 20-year buckets ----
    start = (years.min() // 20) * 20
    end = ((years.max() // 20) + 1) * 20

    print("\n🗂️ 20-year buckets:")
    for y in range(start, end, 20):
        count = ((years >= y) & (years < y + 20)).sum()
        if count == 0:
            continue
        print(f"   {y}-{y+19}: {count:5d} ({count / total * 100:5.1f}%)")

    # ---- Quantiles ----
    print("\n📈 Quantiles:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"   {int(q*100):>2}%: {int(years.quantile(q))}")

    # --------------------------------------------------
    # Store final datasets
    # --------------------------------------------------
    ratings_out = output_dir / "ratings_reduced.csv"
    movies_out = output_dir / "movies_reduced.csv"

    ratings.to_csv(ratings_out, index=False)
    movies.to_csv(movies_out, index=False)

    print("\n💾 Saved:")
    print(f"   {ratings_out}")
    print(f"   {movies_out}")
    print("✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce MovieLens dataset to head data with recency weighting"
    )

    parser.add_argument("--input-dir", type=str, default=str(ANALYSIS_ROOT / "dataset" / "movies" / "ml-1m"))
    parser.add_argument("--output-dir", type=str, default="./")

    parser.add_argument("--top-n-movies", type=int, default=100000,
                        help="Number of top movies to keep")
    parser.add_argument("--max-ratings", type=int, default=5_000_000,
                        help="Maximum number of ratings to keep (None = keep all)")
    parser.add_argument("--recency-weight", type=float, default=0.3,
                        help="Weight for recency bonus (0=ignore age, 1=strong preference for new)")

    parser.add_argument("--links-filter", type=str, default=SCRIPT_DIR / "links_repaired_filtered.csv",
                        help="Optional CSV with movieId column to restrict dataset (e.g. links_repaired_filtered.csv)")

    parser.add_argument("--min-user-interactions", type=int, default=30)
    parser.add_argument("--min-movie-interactions", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    args = parser.parse_args()
    main(args)