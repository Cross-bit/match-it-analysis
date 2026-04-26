# Unified datasets layout

This repository supports a unified top-level datasets root:

- `dataset/movies/`
- `dataset/restaurants/`

Loaders use only this unified root.

## Recommended target structure

```text
dataset/
  movies/
    ml-1m/
      movies.csv
      ratings.csv
    ml-32m/
      movies.csv
      ratings.csv
      README.txt
      checksums.txt
  restaurants/
    places.json
```

Legacy dataset paths under `movies_data/dataset/` and `restaurant_data/dataset/` are deprecated.

## Restaurant data ingest source

Restaurant data collection scripts were intentionally removed from this repository.
Use the dedicated tooling in the application repository (`match-it-demo-backend/tools/data`)
to fetch/update restaurant source data, then place final JSON outputs into:

- `dataset/restaurants/places.json`
- `dataset/restaurants/_places.json` (optional backup/intermediate file)
