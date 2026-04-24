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
