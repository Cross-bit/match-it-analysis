#!/usr/bin/env python3
#remove_empty_ids.py

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def remove_empty_tmdb_ids(
    input_csv: Path,
    output_csv: Path,
    dedupe: bool = False,
):
    kept = 0
    removed_empty = 0
    removed_dupe = 0

    seen_tmdb_ids: set[str] = set()

    with input_csv.open(encoding="utf-8", newline="") as src, \
        output_csv.open("w", encoding="utf-8", newline="") as dst:

        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            tmdb_id = row.get("tmdbId")

            # 1️⃣ remove empty / null tmdbId
            if tmdb_id is None or tmdb_id.strip() == "":
                removed_empty += 1
                continue

            tmdb_id = tmdb_id.strip()

            # 2️⃣ optional deduplication
            if dedupe:
                if tmdb_id in seen_tmdb_ids:
                    removed_dupe += 1
                    continue
                seen_tmdb_ids.add(tmdb_id)

            writer.writerow(row)
            kept += 1

    print("===================================")
    print(f"Kept rows:           {kept}")
    print(f"Removed empty tmdbId:{removed_empty}")
    if dedupe:
        print(f"Removed duplicates: {removed_dupe}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Remove empty tmdbId rows and optionally deduplicate tmdbId"
    )
    parser.add_argument(
        "--links",
        help="Path to original links.csv",
        default=SCRIPT_DIR / "links_original.csv",
    )
    parser.add_argument(
        "--out",
        help="Path to output links.csv",
        default=SCRIPT_DIR / "links.csv",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove rows with duplicate tmdbId (keeps first occurrence)",
    )

    args = parser.parse_args()

    remove_empty_tmdb_ids(
        args.links,
        args.out,
        dedupe=args.dedupe,
    )
