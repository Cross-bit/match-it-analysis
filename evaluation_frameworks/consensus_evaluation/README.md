# Consensus evaluation package

This package implements the **group consensus** experiment code used with MovieLens-style pipelines: mediators (async / sync / hybrid), round-based user feedback simulation, and evaluation scripts that produce metrics and LaTeX-friendly tables.

For **how to run** experiments (venv, dataset layout, `make` targets, environment variables), see the repository root [README.md](../../README.md).

**Single experiments** (one `eval_*` / `tune_*` module, tables, larger-group scripts) are usually started from the repo root via the root **`Makefile`** (e.g. `make tune_hybrid_all_params`, `make consensus-eval-dataset-gen`).

The sibling package **`general_recommender_evaluation/`** holds separate MovieLens / Surprise tooling; it is outside this consensus package.

## Layout

| Path | Role |
|------|------|
| `consensus_mediator.py` | Mediator orchestration and threshold policies (package entry point used by evaluators). |
| `consensus_algorithm/` | Core algorithm pieces: `recommender_engine`, `redistribution_unit`, `priority_queue`, `models`. |
| `synthetic_groups/` | Synthetic group construction, test-set splits, embedding helpers, generator tests. |
| `evaluation/` | Experiment pipeline: dataset preparation, shared config, and runnable experiment modules under `evaluations/`. |
| `evaluation/evaluations/evaluators/` | Evaluator stack (`Runner`, factories, `ConsensusAgentBasedEvaluator`, agents); package overview in `evaluators/__init__.py`. |

### Why `evaluation` / `evaluations`?

The outer **`evaluation/`** package is the whole “run experiments on data” slice (preparation, factories, orchestration helpers). The inner **`evaluations/`** folder is only the **entrypoint modules** (`eval_*`, `tune_*`, `print/`, `larger_group_evaluations/`, …). The names overlap on purpose in English (“evaluation of evaluation”) but it is a bit nested; renaming the inner folder (for example to **`experiment_runs`**, **`cases`**, or **`suites`**) would be clearer, but it is a **large** mechanical change (imports, `Makefile`, and every `-m …evaluation.evaluations…` path).

Nested README (dataset paths, preparation):

- [evaluation/evaluation_preparation/README.md](evaluation/evaluation_preparation/README.md)
