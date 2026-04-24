#!/usr/bin/env python3
#filter_repaired_ids.py

import csv
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# =====================================================
# CONFIG
# =====================================================

SIMILARITY_THRESHOLD = 0.65
SCRIPT_DIR = Path(__file__).resolve().parent

# =====================================================
# HELPERS
# =====================================================

def normalize_title(title: str) -> str:
    """
    Normalize movie titles for comparison.
    - lowercase
    - remove year "(1995)"
    - remove punctuation
    """
    title = title.lower()
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    return title.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def extract_year_from_movielens(title: str) -> int | None:
    """
    Extract year from MovieLens title: "Heat (1995)" -> 1995
    """
    m = re.search(r"\((\d{4})\)", title)
    if m:
        return int(m.group(1))
    return None


def extract_year_from_release_date(date_str: str | None) -> int | None:
    """
    Extract year from TMDB release_date: "1995-12-15" -> 1995
    """
    if not date_str:
        return None
    if len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None


# =====================================================
# LOADERS
# =====================================================

def load_movies(path: Path) -> dict[int, str]:
    movie_titles = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["movieId"]
            title = row["title"]
            if mid.isdigit():
                movie_titles[int(mid)] = title
    return movie_titles


def load_links(path: Path) -> dict[int, int]:
    """
    tmdbId -> movieId
    """
    mapping = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["movieId"]
            tid = row["tmdbId"]
            if mid.isdigit() and tid.isdigit():
                mapping[int(tid)] = int(mid)
    return mapping


def load_repaired(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["old_tmdb_id"] = int(row["old_tmdb_id"])
            row["new_tmdb_id"] = int(row["new_tmdb_id"])
            row["popularity"] = float(row.get("popularity", 0))
            rows.append(row)
    return rows


# =====================================================
# CORE
# =====================================================

def main(repaired_csv, links_csv, movies_csv, out_csv):
    movies = load_movies(movies_csv)
    links = load_links(links_csv)
    repaired = load_repaired(repaired_csv)

    # -------------------------------------------------
    # 1️⃣ Remove duplicates in repaired.csv
    # keep highest popularity per old_tmdb_id
    # -------------------------------------------------

    dedup = {}
    for r in repaired:
        key = r["new_tmdb_id"]
        if key not in dedup or r["popularity"] > dedup[key]["popularity"]:
            dedup[key] = r

    repaired = list(dedup.values())

    # -------------------------------------------------
    # 2️⃣ Merge + similarity filter
    # -------------------------------------------------

    kept = []
    dropped = 0

    for r in repaired:
        old_tmdb = r["old_tmdb_id"]

        movie_id = links.get(old_tmdb)
        if not movie_id:
            dropped += 1
            continue

        ml_title = movies.get(movie_id)
        if not ml_title:
            dropped += 1
            continue

        a = normalize_title(ml_title)
        b = normalize_title(r["matched_title"])

        score = similarity(a, b)

        # year consistency check
        ml_year = extract_year_from_movielens(ml_title)
        tmdb_year = extract_year_from_release_date(r.get("release_date"))

        if ml_year and tmdb_year and ml_year != tmdb_year:
            dropped += 1
            continue

        if score >= SIMILARITY_THRESHOLD:
            r["movieId"] = movie_id
            r["movielens_title"] = ml_title
            r["title_similarity"] = round(score, 3)
            kept.append(r)
        else:
            dropped += 1


    # -------------------------------------------------
    # 3️⃣ Save
    # -------------------------------------------------

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "movieId",
            "old_tmdb_id",
            "new_tmdb_id",
            "movielens_title",
            "matched_title",
            "title_similarity",
            "release_date",
            "popularity",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in kept:
            writer.writerow({
                k: row.get(k)
                for k in fieldnames
            })

    print("===================================")
    print(f"Kept rows:   {len(kept)}")
    print(f"Dropped:     {dropped}")
    print(f"Output:      {out_csv}")


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate repaired TMDB IDs using title similarity"
    )

    parser.add_argument(
        "--repaired",
        help="Path to repaired.csv",
        default=SCRIPT_DIR / "repaired.csv",
    )
    parser.add_argument(
        "--links",
        help="Path to links.csv",
        default=SCRIPT_DIR / "links.csv",
    )
    parser.add_argument(
        "--movies",
        help="Path to movies.csv",
        default=SCRIPT_DIR / "movies.csv",
    )
    parser.add_argument(
        "--out",
        help="Output CSV",
        default=SCRIPT_DIR / "repaired_filtered.csv",
    )

    args = parser.parse_args()

    main(
        Path(args.repaired),
        Path(args.links),
        Path(args.movies),
        Path(args.out),
    )
