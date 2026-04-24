# Analysis Directory

This is primarily a Python project designed to test and analyze data for a recommendation system.

## Execution

💡 Use the attached *Makefile* in the root directory to run various evaluations.
Refer to the file for details on the available commands.

For example:

```
make restaurant-algo-comparison
```

## Output

The typical output is a chart plotted using the *matplotlib.pyplot* library.

⚙️ **Note**: Some systems may require additional configuration or script adjustments.
This setup has been tested on Windows 11 with WSL.

📌 **Important**: Since the analysis is intended for use in a *LaTeX* project, the scripts automatically attempt to save generated images as **.pdf** files in the parent directory's **img/** folder.
Ensure this directory exists, **or** update the `IMG_OUTPUT_PATH` constant in `utils/config.py`.

📌 **Caching**: All cached outputs should go to the shared directory `analysis/cache/`.
Use `utils.config` helpers (`load_or_build_pickle`, `load_from_pickle`, `save_to_pickle`, `cache_path`) and avoid storing `.pkl` files next to scripts.

### Cache Layout (organized namespaces)

```text
cache/
  cons_evaluations/                 # consensus evaluation outputs
  movies/
    init-sampling/                  # init sampling outputs and summaries
      kmeans/                       # kmeans-related caches
    algorithm/
      easer/                        # EASER tuning/eval caches
    links-filtering/                # mapping/cleaning/dedup related caches
  restaurants/
    hybrid-algorithm/               # restaurant algorithm sweeps and related caches
      easer/
      item-knn/
      knn/
      sparsity/
```

When adding a new cache file, place it under the matching domain branch (`movies/...`, `restaurants/...`, or `cons_evaluations/...`) so it is obvious what produced it.
Avoid writing unnamed files directly under `cache/`.

For detailed consensus results layout, see:
`evaluation_frameworks/consensus_evaluation/evaluation/evaluations/RESULT_CACHE_LAYOUT.md`

---

Some scripts may output only LaTeX-formatted tables or other statistics directly to the standard output.

---

## Directory Structure

```
project-root/
├── evaluation_framework/    # Evaluation framework and basic recommendation algorithms
├── latex_utils/             # Helper functions for LaTeX generation (e.g., table creation)
├── math_analysis/           # Mathematical function analysis
├── movies_data/             # Analysis of the MovieLens 1M dataset
├── restaurant_data/         # Analysis of restaurant data fetched using Google APIs
├── utils/                   # General utilities (e.g., configuration)
└── orchestration/           # (Not supported yet) Scripts for automated text editing
```