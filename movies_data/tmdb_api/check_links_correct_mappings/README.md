# Attempt to resolve the missing movie lens id to tmdb ids mappings

**Expected execution order**:

0. `remove_empty_ids.py && remove_duplicates.py` -- remove movies with missing tmdb id and duplicate returns (about ~126 records) from links_original.csv (original links from ml32)
1. `validated_tmdb_ids.py` -- check all tmdb ids in links.csv creates list of ids which en content is unresolvable and outputs it to missing_en_tmdb_ids.csv
2. `Find missing_tmdb_ids.py` -- takes missing_en_tmdb_ids.csv, outputs repaired.csv, unresolved.csv
3. `filter_repaired_ids.py` -- takes repaired.csv and removes duplicit new tmdb ids and other inconsistent results
4. `resolve_missing_tmdb_ids.py` -- takes repaired_filtered.csv, unresolved.csv and original links.csv, outputs links_repaired with new ids from the repaired.csv
5. `remove_duplicates_tmdb_ids.py` -- removes duplicit tmdb ids from the links file. (Even in the original some tmdb movies had duplicates)