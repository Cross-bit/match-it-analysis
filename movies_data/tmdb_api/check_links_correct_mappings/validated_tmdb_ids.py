#!/bin/python3
#validate_tmdb_ids.py

import csv
import os
import time
import logging
import argparse
from pathlib import Path
import requests
from typing import List, Dict

from tqdm import tqdm

# =====================================================
# DESCRIPTION
# =====================================================
# Loads all list of unresolvable TMDB ids from missing_en_tmdb_ids.csv
# and tries to resolve current movie id from the redirected URL and
# TMDB movie search API using this obtained slug.
#
# Output:
# repaired.csv -- old_tmdb_id,new_tmdb_id,query_title,matched_title,release_date,popularity
#   contains all the new mappings for all the old movies.
#
# unresolved.csv -- old_tmdb_id,query_title,reason
#   list of unresolvable tmdb ids, with searched query and the reason of failure
#
#

TMDB_BASE_URL = "https://api.themoviedb.org/3"
RATE_LIMIT_DELAY = 0.25  # ~4 req/s (bezpečné)
TIMEOUT = 10

SCRIPT_DIR = Path(__file__).resolve().parent
#print(SCRIPT_DIR)

def load_tmdb_ids(csv_path: str) -> List[int]:
    tmdb_ids = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tmdb_id = row.get("tmdbId")
            if tmdb_id and tmdb_id.strip().isdigit():
                tmdb_ids.append(int(tmdb_id))

    return tmdb_ids


def fetch_movie_en(tmdb_id: int, api_key: str) -> Dict | None:
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "append_to_response": "credits",
    }

    resp = requests.get(url, params=params, timeout=TIMEOUT)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    return resp.json()


def main(csv_path: str, api_key: str, output_path: str):
    tmdb_ids = load_tmdb_ids(csv_path)

    logging.info(f"Loaded {len(tmdb_ids)} TMDB IDs")

    missing_en = []
    ok = 0
    errors = 0

    for idx, tmdb_id in enumerate( tqdm(tmdb_ids, desc="Checking TMDB movies", unit="movie"), 1):
        try:
            data = fetch_movie_en(tmdb_id, api_key)

            if not data or not data.get("title"):
                logging.warning(f"[MISSING EN] TMDB ID {tmdb_id}")
                missing_en.append(tmdb_id)
            else:
                ok += 1

        except requests.HTTPError as e:
            logging.error(f"[HTTP ERROR] TMDB ID {tmdb_id}: {e}")
            errors += 1
            missing_en.append(tmdb_id)

        except Exception as e:
            logging.error(f"[ERROR] TMDB ID {tmdb_id}: {e}")
            errors += 1
            missing_en.append(tmdb_id)

        if idx % 50 == 0:
            logging.info(f"Processed {idx}/{len(tmdb_ids)}")

        time.sleep(RATE_LIMIT_DELAY)

    # Save bad IDs
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tmdbId"])
        for tid in missing_en:
            writer.writerow([tid])

    logging.info("===================================")
    logging.info(f"OK movies: {ok}")
    logging.info(f"Missing EN / invalid: {len(missing_en)}")
    logging.info(f"Errors: {errors}")
    logging.info(f"Saved to: {output_path}")


if __name__ == "__main__":

    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

    parser = argparse.ArgumentParser(description="Validate TMDB IDs from MovieLens CSV")
    parser.add_argument("--csv", help="Path to MovieLens links.csv", default=SCRIPT_DIR / "links.csv")
    parser.add_argument("--api-key", help="TMDB API key", default=TMDB_API_KEY)
    parser.add_argument(
        "--out",
        default=SCRIPT_DIR / "missing_en_tmdb_ids.csv",
        help="Output CSV with invalid TMDB IDs",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    main(args.csv, args.api_key, args.out)
