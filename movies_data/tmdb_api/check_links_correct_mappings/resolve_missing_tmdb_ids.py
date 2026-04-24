#!/bin/python3
#resolve_missing_tmdb_ids.py

import csv
import argparse
import logging
from pathlib import Path

# =====================================================
# DESCRIPTION
# =====================================================
# Using generated repaired.csv tries to repair incorrect TMDB ids in the links.csv file
#
# Optional -- remove incorrect ids from the unresolved.csv.
#
# Output:
# links_repaired.csv -- movieId,imdbId,tmdbId
#   contains all links with repaired ids.

SCRIPT_DIR = Path(__file__).resolve().parent


# =====================================================
# HELPERS
# =====================================================

def normalize_int(value: str | None) -> int | None:
    """
    Normalize MovieLens / TMDB numeric values.
    Handles: "123", "123.0", " 123 ", None
    """
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


# =====================================================
# LOADERS
# =====================================================

def load_repaired(path: Path) -> dict[int, int]:
    """
    Returns mapping:
    old_tmdb_id -> new_tmdb_id
    """
    repaired: dict[int, int] = {}

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"old_tmdb_id", "new_tmdb_id"}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"repaired CSV must contain columns {required_cols}, "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            old = normalize_int(row.get("old_tmdb_id"))
            new = normalize_int(row.get("new_tmdb_id"))

            if old is None or new is None:
                continue

            repaired[old] = new

    logging.info(f"Loaded {len(repaired)} repaired TMDB ID mappings")
    return repaired


def load_unresolved(path: Path) -> set[int]:
    unresolved = set()

    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                tid = normalize_int(row[0])
                if tid is not None:
                    unresolved.add(tid)

    logging.info(f"Loaded {len(unresolved)} unresolved TMDB IDs")
    return unresolved


# =====================================================
# CORE LOGIC
# =====================================================

def process_links(
    links_path: Path,
    repaired_path: Path,
    unresolved_path: Path,
    output_path: Path,
    mode: str,
):
    repaired = load_repaired(repaired_path)
    unresolved = load_unresolved(unresolved_path)

    kept = 0
    dropped = 0
    fixed = 0

    with links_path.open(encoding="utf-8", newline="") as src, \
        output_path.open("w", encoding="utf-8", newline="") as dst:

        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            movie_id = normalize_int(row.get("movieId"))
            tmdb_id = normalize_int(row.get("tmdbId"))

            if tmdb_id is None:
                writer.writerow(row)
                kept += 1
                continue

            if tmdb_id in repaired:
                row["tmdbId"] = str(repaired[tmdb_id])
                writer.writerow(row)
                fixed += 1
                continue

            if tmdb_id in unresolved:
                if mode == "drop-unresolved":
                    dropped += 1
                    continue

            writer.writerow(row)
            kept += 1

    logging.info("===================================")
    logging.info(f"Fixed rows: {fixed}")
    logging.info(f"Kept rows: {kept}")
    logging.info(f"Dropped rows: {dropped}")
    logging.info(f"Output: {output_path}")


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge links.csv with repaired & unresolved TMDB IDs")
    parser.add_argument("--links", help="Path to original links.csv", default=SCRIPT_DIR / "links.csv")
    parser.add_argument("--repaired", help="Path to repaired.csv", default=SCRIPT_DIR / "repaired_filtered.csv")
    parser.add_argument("--unresolved", help="Path to unresolved.csv", default=SCRIPT_DIR / "unresolved.csv")
    parser.add_argument(
        "--mode",
        choices=["repaired", "drop-unresolved"],
        default="drop-unresolved",
        help="How to handle unresolved TMDB IDs (default: repaired)"
    )
    parser.add_argument(
        "--out",
        default=SCRIPT_DIR / "links_repaired.csv",
        help="Output CSV"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    process_links(
        Path(args.links),
        Path(args.repaired),
        Path(args.unresolved),
        Path(args.out),
        args.mode,
    )
