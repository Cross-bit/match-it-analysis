# Analysis Workspace

Analytická část projektu pro evaluaci doporučovacích algoritmů (movies + restaurants + consensus experiments).

## Quick Start

Spouštění je standardně přes `Makefile`:

```bash
make restaurant-algo-comparison
```

Výstupy jsou typicky:
- grafy uložené do `img/` (většinou `.pdf`),
- LaTeX tabulky vypsané do stdout,
- cache artefakty v `cache/`.

## Nejčastější příkazy

### Restaurant

```bash
# CF porovnání algoritmů (EASE, Popularity, ItemKNN, UserKNN, SVD)
make restaurant-algo-comparison

# totéž pro jiný filtr uživatelů (r_min)
make CB_MIN_USER_RATING=3 restaurant-algo-comparison

# CB evaluace
make restaurant-cb-evaluation
make CB_MIN_USER_RATING=4 CB_K=20 CB_TRAIN_RATIO=0.8 restaurant-cb-evaluation

# KNN sousedi (graf G.2)
make restaurant-knn-neighbors-by-test-size

# Item overlap (graf G.1)
make restaurant-overlaps-by-test-size

# EASE lambda sweep (tabulka G.1)
make restaurant-optimal-easer-lambda
make MODE=compute EASER_LAMBDAS="100 200 400 800 1600 3200 6400 12800 25600" restaurant-optimal-easer-lambda
```

### Consensus

```bash
make consensus-eval-dataset-gen
make tune_sync_with_feedback_ema
make eval_sync_with_feedback_ema
make plot_success_rate_by_bias
```

### Movies

```bash
make movie-dataset-stats
make movie-dataset-genre-frequencies
make movie-popularity-histogram
make movie-knn-algo-comparison
```

## MODE přepínač (cache workflow)

Vybrané skripty podporují:
- `MODE=auto` (default): load cache, jinak compute + save
- `MODE=load`: pouze načíst cache
- `MODE=compute`: vždy přepočítat a přepsat cache

Příklad:

```bash
make MODE=compute restaurant-algo-comparison
```

## Cache pravidla

Všechny průběžné artefakty mají být v `cache/`, ne v rootu projektu.

Používej helpery z `utils/config.py`:
- `load_or_build_pickle`
- `load_from_pickle`
- `save_to_pickle`
- `cache_path`

Doporučené namespace větve:

```text
cache/
  cons_evaluations/
  movies/
  restaurants/
```

## Rejstřík Skriptů (CZ, pro elektronickou přílohu)

### Restaurant – hlavní
- `restaurant_data/algo_experiments/comparison_of_algorithms.py`  
  CF porovnání algoritmů, výstupem je LaTeX tabulka.
- `restaurant_data/algo_experiments/cb_model_precision.py`  
  Evaluace `Simple CB`, výstupem je LaTeX tabulka.
- `restaurant_data/algo_experiments/knn_neighbors_count_by_test_size.py`  
  Analýza průměrného počtu sousedů (`UserKNN` vs `ItemKNN`), výstup graf.
- `restaurant_data/sparsity_issue/overlaps_by_dataset_size.py`  
  Analýza překryvů položek dle filtru uživatelů, výstup graf.
- `restaurant_data/algo_experiments/optimal_easer_lambda.py`  
  Sweep regularizace `lambda` pro `EASE^R`, výstup tabulka.

### Consensus – hlavní
- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/tune_*.py`  
  Ladění parametrů variant konsensu.
- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/eval_*.py`  
  Finální evaluace variant.
- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/print/*.py`  
  Exporty tabulek/grafů pro text práce.

### Movies – hlavní
- `movies_data/algo_experiments/comparison_of_algorithms.py`
- `movies_data/genres_coverage_frequencies.py`
- `movies_data/popularity_histogram.py`
- `movies_data/production_filtering/filter_dataset.py`

## Poznámky k čistotě

- Lokální helper shell skripty (`eval_notify.sh`, `export_full_rfc_csv.sh`, `get_large_grps.sh`) nejsou součástí verzované části.
- Root `.pkl` artefakty nejsou povoleny (patří do `cache/`).