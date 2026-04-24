#!/bin/python3
#find_missing_tmdb_ids.py

import os
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import requests
from tqdm import tqdm


# =====================================================
# DESCRIPTION
# =====================================================
# Goes through all movies in provided links.csv that includes
# mapping of MovieLens ids to TMDB IDs
# and tries to fetch the en content from the TMDB movies
# database.
#
# Check the entrypoint for required params. TMDB API key can be provided using env TMDB_API_KEY.
#
# OUTPUT:
# missing_en_tmdb_ids.csv -- a single col of TMDB ids for which
# the en description could not be resolved.
#


TMDB_BASE_URL = "https://api.themoviedb.org/3" # base API url
TMDB_WEB_MOVIE_URL = "https://www.themoviedb.org/movie"
RATE_LIMIT_DELAY = 0.15 # rate limit between requests
TIMEOUT = 10

SCRIPT_DIR = Path(__file__).resolve().parent


# ----------------------------
# Utils
# ----------------------------

def slug_to_title(slug: str) -> str:
    """
    d-b-cooper-where-are-you -> D B Cooper Where Are You
    """
    return slug.replace("-", " ").strip()


def extract_slug_from_url(url: str) -> Optional[str]:
    """
    https://www.themoviedb.org/movie/1005682-d-b-cooper-where-are-you
    -> d-b-cooper-where-are-you
    """
    try:
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        if "-" not in tail:
            return None
        return tail.split("-", 1)[1]
    except Exception:
        return None


# ----------------------------
# TMDB logic
# ----------------------------

def follow_tmdb_redirect(tmdb_id: int) -> Optional[str]:
    """
    Returns slug if redirect exists, otherwise None
    """
    url = f"{TMDB_WEB_MOVIE_URL}/{tmdb_id}"
    resp = requests.get(url, allow_redirects=False, timeout=TIMEOUT)

    if resp.status_code >= 400:
        return None

    if resp.is_redirect:
        new_location = resp.headers['Location']
        return extract_slug_from_url(new_location)





def search_movie(api_key: str, title: str) -> List[Dict]:
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": api_key,
        "query": title,
        "language": "en-US",
        "include_adult": False,
    }

    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("results", [])


def choose_best_match(results: List[Dict], query_title: str) -> Optional[Dict]:
    """
    Conservative selection:
    - exact (case-insensitive) title match OR
    - highest popularity if only one result
    """
    if not results:
        return None

    query_norm = query_title.lower()

    exact = [
        r for r in results
        if r.get("title", "").lower() == query_norm
    ]

    if exact:
        return exact[0]

    if len(results) == 1:
        return results[0]

    # fallback: highest popularity
    results.sort(key=lambda r: r.get("popularity", 0), reverse=True)
    return results[0]


# ----------------------------
# CSV helpers
# ----------------------------

def load_tmdb_ids(csv_path: Path) -> List[int]:
    ids = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("tmdbId")
            if tid and tid.isdigit():
                ids.append(int(tid))
    return ids


# ----------------------------
# Main
# ----------------------------

def main(
    input_csv: Path,
    api_key: str,
    repaired_csv: Path,
    unresolved_csv: Path,
):
    tmdb_ids = load_tmdb_ids(input_csv)

    repaired = []
    unresolved = []

    for tmdb_id in tqdm(tmdb_ids, desc="Repairing TMDB IDs", unit="movie"):
        try:
            slug = follow_tmdb_redirect(tmdb_id)

            if not slug:
                unresolved.append((tmdb_id, "", "no_redirect"))
                continue

            title = slug_to_title(slug)
            results = search_movie(api_key, title)
            best = choose_best_match(results, title)

            if not best:
                unresolved.append((tmdb_id, title, "no_search_match"))
                continue

            repaired.append((
                tmdb_id,
                best["id"],
                title,
                best.get("title", ""),
                best.get("release_date", ""),
                best.get("popularity", 0),
            ))

        except Exception as e:
            unresolved.append((tmdb_id, "", str(e)))

        time.sleep(RATE_LIMIT_DELAY)

    # save repaired
    with open(repaired_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "old_tmdb_id",
            "new_tmdb_id",
            "query_title",
            "matched_title",
            "release_date",
            "popularity",
        ])
        writer.writerows(repaired)

    # save unresolved
    with open(unresolved_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "old_tmdb_id",
            "query_title",
            "reason",
        ])
        writer.writerows(unresolved)

    logging.info(f"Repaired: {len(repaired)}")
    logging.info(f"Unresolved: {len(unresolved)}")


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    import argparse

    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")


    parser = argparse.ArgumentParser(description="Repair missing TMDB IDs")
    parser.add_argument("--input", help="CSV with tmdbId column", default=SCRIPT_DIR / "missing_en_tmdb_ids.csv")
    parser.add_argument("--api-key", help="TMDB API key", default=TMDB_API_KEY)
    parser.add_argument("--out-ok", default=SCRIPT_DIR / "repaired.csv")
    parser.add_argument("--out-bad", default=SCRIPT_DIR / "unresolved.csv")

    args = parser.parse_args()

    main(
        Path(args.input),
        args.api_key,
        Path(args.out_ok),
        Path(args.out_bad),
    )
