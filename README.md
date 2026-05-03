# Analysis Workspace

Analytická část práce — skripty a experimenty k doporučování. V repozitáři jsou zhruba **tři oblasti**:

1. **Movies** — experimenty nad MovieLens (long-tail, žánry, inicializace vzorků, k-means, srovnání CF …) a úpravy datasetu: `movies_data/production_filtering/`, podprojekt MovieLens ↔ TMDB (`movies_data/tmdb_api/check_links_correct_mappings/`, dokumentace v jeho `README.md`).
2. **Restaurants** — stejné typy úloh nad daty z Google Places (řídkost, pokrytí, CF/CB tabulky, EASE a pod.).
3. **Consensus experiments** — evaluace **hlavních algoritmů práce**: simulované skupiny uživatelů, varianty sync/async konsenzu, ladění parametrů a export tabulek/grafů do textu.

Oblasti sdílejí společné helpery (`utils/`), `Makefile` a konvenci ukládání výstupů (`img/`, `cache/`).

## Quick Start

Spouštění je standardně přes `Makefile`:

```bash
make restaurant-algo-comparison
```

Výstupy jsou typicky:

- grafy uložené do `img/` (většinou `.pdf`),
- LaTeX tabulky vypsané do stdout,
- cache artefakty v `cache/`.

Ukázka grafu do `img/`: `make movie-popularity-histogram` → soubor `long_tail_popularity_movielens.pdf` (MovieLens musí být pod `dataset/movies/…`).

Na prostředí bez okna (typicky WSL bez X11) může Matplotlib u `plt.show()` čekat na GUI a proces zamrzne. V `Makefile` je proto `export MPLBACKEND=Agg`: `**MPLBACKEND**` je oficiální proměnná prostředí Matplotlibu — nastavuje backend; hodnota `Agg` kreslí jen do souboru (raster/PDF), bez interaktivního okna ([proměnné prostředí](https://matplotlib.org/stable/installing/environment_variables.html)). Při ručním `python3 -m …` mimo `make` uvést např. `MPLBACKEND=Agg python3 -m movies_data.popularity_histogram`.

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
make groups-generator
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

Proměnná `MODE` z `Makefile` (`auto` / `load` / `compute`) se propisuje jen do **těch cílů**, které ji v příkazu předávají — především evaluace/tuning konsenzu a vybrané restaurant skripty (např. `restaurant-algo-comparison`, `restaurant-cb-evaluation`, `restaurant-knn-neighbors-by-test-size`, `restaurant-optimal-easer-lambda`). U ostatních `make` příkazů se typicky neprojeví.

Tam, kde se použije, platí:

- `MODE=auto` (default): načíst cache, jinak přepočítat a uložit
- `MODE=load`: pouze načíst cache
- `MODE=compute`: vždy přepočítat a přepsat cache

Příklad (restaurant CF tabulka):

```bash
make MODE=compute restaurant-algo-comparison
```

## Cache pravidla

Průběžné artefakty (pickle, mezipočty apod.) se ukládají pod `cache/`.

V `utils/config.py` jsou k dispozici například:

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

**Dataset restaurací:** surová data patří do `dataset/restaurants/` (typicky `places.json`, v gitu často ignorované). **Odvozené** drobné tabulky ze skriptů (např. `user_reviews_percentages.csv`) ukládat do `dataset/restaurants/derived/`, nikoli do `restaurant_data/`.

## Mapování práce (LaTeX) ↔ výstupy z tohoto repozitáře

PDF grafy z analytických skriptů jdou do `analysis/img/` (stejná úroveň jako `analysis/cache/`; `IMG_OUTPUT_PATH` v `utils/config.py`). Složka `img/` se při importu `utils.config` založí, pokud chybí.

**Tabulky** skript obvykle vytiskne jako LaTeX na **stdout** (zkopírovat do `.tex`).

**Spuštění end-to-end** vyžaduje lokální data (MovieLens, `dataset/restaurants/…`, cache evaluací u konsenzu) — bez nich skript spadne na načtení, i když je syntax OK.

### Grafy (PDF) — citované v textu / příloze, generované zde


| Soubor v `img/`                                                                                                                    | Kde v práci (orientační)                                                             | Skript                                                                 | Spuštění                                        |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------- |
| `item-overlap-counts.pdf`                                                                                                          | příloha (`thesis.tex`), překryvy položek                                             | `restaurant_data/sparsity_issue/overlaps_by_dataset_size.py`           | `make restaurant-overlaps-by-test-size`         |
| `knn-neighbors-avg-count.pdf`                                                                                                      | příloha, User vs Item KNN sousedi                                                    | `restaurant_data/algo_experiments/knn_neighbors_count_by_test_size.py` | `make restaurant-knn-neighbors-by-test-size`    |
| `knn-evaluation-by-dataset-size.pdf`                                                                                               | (volitelný experiment)                                                               | `restaurant_data/algo_experiments/knn_by_dataset_size.py`              | `make restaurant-knn-by-test-size`              |
| `histogram-of-user-reviews-all.pdf`                                                                                                | frekvence recenzí uživatelů                                                          | `restaurant_data/places_coverage/users_ratings_frequencies.py`         | `make restaurant-reviews-histogram`             |
| `user-place-coverage-histogram.pdf`                                                                                                | pokrytí míst podle top‑n uživatelů                                                   | `restaurant_data/places_coverage/cumulative_coverage.py`               | `make restaurant-places-coverage`               |
| `places-users-coverage.pdf`                                                                                                        | pokrytí + lidé                                                                       | `restaurant_data/places_coverage/cumulative_coverage_with_people.py`   | `make restaurant-places-coverage-people`        |
| `density-histograms.pdf`, `density-only-histogram.pdf`, `density-histograms-user-ratings-only.pdf`, `density-histograms-lines.pdf` | řídkost / hustota po řezu datasetem                                                  | `restaurant_data/sparsity_issue/sparsity_users_elimination_method.py`  | `make restaurant-sparsity-dataset`              |
| `long_tail_popularity_movielens.pdf`                                                                                               | MovieLens long-tail                                                                  | `movies_data/popularity_histogram.py`                                  | `make movie-popularity-histogram`               |
| `genres_distribution.pdf`                                                                                                          | žánry                                                                                | `movies_data/genres_coverage_frequencies.py`                           | `make movie-dataset-genre-frequencies`          |
| `movie_popularity_by_release_year.pdf`                                                                                             | **součet počtu ratingů** v ML podle **roku v názvu filmu** (ne průměr hvězdiček)     | `movies_data/release_year_popularity_histogram.py`                     | `make movie-release-year-popularity`            |
| `clustering-kmeans-elbow-method.pdf`, `clustering-silueth-score.pdf`                                                               | elbow + silueta při volbě **k** pro k-means nad reprezentacemi filmů (init sampling) | `movies_data/initialisation_sampling/clustering/kmeans_analysis.py`    | `make kmeans-analysis`                          |
| `init-sampling-comparison.pdf`                                                                                                     | srovnání strategií výběru filmů                                                      | `movies_data/initialisation_sampling/init_sampling_test_all.py`        | `make init-sampling-all`                        |
| `init_sample_size_accuracy2_popularity_comp.pdf`                                                                                   | RMSE vs velikost vzorku                                                              | `movies_data/initialisation_size.py`                                   | `make init-size`                                |
| `recommendation_year_distribution_ease_multiplicative_beta_*.pdf` (název podle `RERANK_MODE` / `RECENCY_PARAM` v souboru)          | rozložení roku po reranku                                                            | `movies_data/distribution_year_release_vs_easer.py`                    | `make movie-distribution-year-release-vs-easer` |


**Není generováno skripty z tohoto repozitáře** (typicky nástroj na diagramy, šablona, nebo jiný projekt): např. `sequel-diversity-importance.pdf`, `ndcg.pdf`, `sigmoid_c0_lower_bound.pdf`, `sigmoid_c0_lower_upper_bound.pdf`, `sigmoids-eval-agent.pdf`, diagramy async/sync konsenzu (`async-*.pdf`, `sync-*.pdf`), UML (`*_UML.pdf`), ER/db (`db_*.pdf`), Android (`ActivitiesTransitions.pdf`, …), `profile_creation.pdf`, `logo-cs.pdf`. Ty patří do přílohy jako zdroje z návrhu/GUI, ne z `analysis/*.py`.

### Tabulky (LaTeX na stdout)


| Účel / label v práci (typicky)                            | Skript                                                                                            | Spuštění                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| CF restaurace (`tab:AlgosComparisionTextTable` apod.)     | `restaurant_data/algo_experiments/comparison_of_algorithms.py`                                    | `make restaurant-algo-comparison`                                              |
| Simple CB                                                 | `restaurant_data/algo_experiments/cb_model_precision.py`                                          | `make restaurant-cb-evaluation`                                                |
| Sweep λ EASE^R (`tab:EaserRegMeasure`)                    | `restaurant_data/algo_experiments/optimal_easer_lambda.py`                                        | `make restaurant-optimal-easer-lambda`                                         |
| Základní statistiky matice hodnocení restaurací           | `restaurant_data/general_analysis.py`                                                             | `make restaurant-general-stats`                                                |
| MovieLens CF srovnání (MovieLens tabulka v kap. evaluace) | `movies_data/algo_experiments/comparison_of_algorithms.py`                                        | `make movie-knn-algo-comparison`                                               |
| Skupiny pro simulaci konsensu (základ dat)                | `evaluation_frameworks/consensus_evaluation/synthetic_groups/groups_generator.py`                 | `make groups-generator`                                                        |
| Ladění / tabulky konsenzu (RFC, NDCG, …)                  | `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/tune_*.py`, `print/table_*.py` | viz Makefile (`make table_ndcg_comparisions`, `make table_rfc_comparision`, …) |


### Jupyter


| Notebook                                          | Poznámka                                                                                                                                                               |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `restaurant_data/restaurants_data_analysis.ipynb` | Průzkumná analýza dat (**EDA**) nad Places; detailněji v rejstříku *Restaurant* níže. Cesta k JSON v notebooku může být jiná než u `DatasetLoader` v Python skriptech. |


---

## Rejstřík Skriptů (CZ, pro elektronickou přílohu)

### Restaurant – hlavní

- `restaurant_data/restaurants_data_analysis.ipynb`  
Průzkum míst (typy, frekvence recenzí, pokrytí top‑n uživateli); očekává lokální JSON a importy z `places_coverage/`.
- `restaurant_data/places_coverage/users_ratings_frequencies.py`  
Histogramy frekvencí; při `python -m …` zapisuje i `dataset/restaurants/derived/user_reviews_percentages.csv`.
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

- `evaluation_frameworks/consensus_evaluation/synthetic_groups/groups_generator.py`  
Generování syntetických **skupin uživatelů** (typicky trojice) pro scénáře evaluace konsensu — embeddingy uživatelů (LightFM), podobnost přes FAISS; artefakty v `cache/`. Dávkové spuštění: `make groups-generator` (`python3 -m evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_generator`).
- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/tune_*.py`  
Ladění parametrů variant konsensu.
- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/eval_*.py`  
Finální evaluace variant.
- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/print/*.py`  
Exporty tabulek/grafů pro text práce.

### Movies – hlavní

- `movies_data/tmdb_api/check_links_correct_mappings/`  
Sjednocení mapování MovieLens ↔ TMDB před produkčním použitím (viz `movies_data/tmdb_api/check_links_correct_mappings/README.md`; vstupy `data/`, výstupy `out/`). CSV, bez pickle.
- `movies_data/production_filtering/filter_dataset.py`  
Úprava MovieLens podle vybraných `movieId` / popularity (výstup do zadaného adresáře).
- `movies_data/algo_experiments/comparison_of_algorithms.py`
- `movies_data/genres_coverage_frequencies.py`
- `movies_data/popularity_histogram.py`

