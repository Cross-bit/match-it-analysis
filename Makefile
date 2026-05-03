# ====================================
# DESCRIPTION
# ====================================
# Analysis execution.
#
#

# Matplotlib bez GUI (WSL bez X11/WSLg, CI). Jinak výchozí backend u plt.show() často zamrzne.
export MPLBACKEND := Agg

# =============================
# GROUP ALGORITHMS
# =============================
groups-generator:
	python3 -m evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_generator

# =============================
# TUNING
# =============================

MODE ?= auto # determines whether to recompute the evaluation options: auto, compute, load
# W: consensus window (--window-size); not related to dataset target below.
W ?= 10

# Run this script to generate the test dataset.
GROUP_SIZE ?= 10
# GROUPS_COUNT: how many groups each eval runs through the simulator (not users-per-group; that is --group-size for large-group targets).
GROUPS_COUNT ?= 1000
MIN_COM ?= 3

consensus-eval-dataset-gen:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation --group-size $(GROUPS_COUNT) --min-com $(MIN_COM)

tune_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_async_with_sigmoid_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_with_sigmoid_policy_simple_priority_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_async_with_static_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_static_policy_simple_priority_function_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_async_with_static_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_static_policy_simple_priority_function_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_without_feedback:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_without_feedback --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_with_feedback_mean:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_with_feedback_mean --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_with_feedback_mean_:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_with_feedback_mean_ --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid_general_rec_individual:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_general_rec_individual --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid: tune_hybrid_general_rec_individual
	@:

tune_hybrid_all_params:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_all_params --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_individual_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

# =============================
# EVALUATIONS
# =============================

eval_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_async_static_policy_simple_priority_function_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_async_static_policy_simple_priority_function_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_sync_without_feedback:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_hybrid_general_rec_individual:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_hybrid_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

# Alias na eval_hybrid_general_rec_individual (sweep first_round_ration → tune_hybrid_general_rec_individual)
eval_hybrid_general_rec_individual_by_first_round_ration:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual_by_first_round_ration --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

# =============================
# LARGER GROUP EVALUATIONS
# =============================

eval_large_hybrid_group_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_large_hybrid_group_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_large_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_async_with_sigmoid_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_async_with_sigmoid_policy_simple_priority_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

# =============================
# EVALUATION SUMMARY
# =============================

table_rfc_comparision:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_comparisions --window-size $(W)

table_rfc_by_population_mood:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood --window-size $(W)

WINDOWS ?= 1 3 5 10
BIASES ?= 0 1 2
table_rfc_by_population_mood_all_windows:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood_all_windows --windows $(WINDOWS) --biases $(BIASES) --groups-count $(GROUPS_COUNT)

table_success_matches_all_windows:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_success_matches_all_windows --windows $(WINDOWS) --biases $(BIASES) --groups-count $(GROUPS_COUNT)

# Success ratio vs bias (matplotlib).
# - default: single window W=5
# - for side-by-side panels set SUCCESS_PLOT_WINDOWS="1 3 5 10"
# - default output format is PDF unless overridden
SUCCESS_PLOT_W ?= 5
SUCCESS_PLOT_WINDOWS ?=
SUCCESS_PLOT_LAYOUT ?= row
SUCCESS_PLOT_PALETTE ?= tab10
SUCCESS_PLOT_FORMAT ?= pdf
SUCCESS_PLOT_ENGLISH ?= 0
SUCCESS_PLOT_GROUP_SIZE ?= 3
plot_success_rate_by_bias:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.plot_success_rate_by_bias $(if $(strip $(SUCCESS_PLOT_WINDOWS)),--windows $(SUCCESS_PLOT_WINDOWS),--window-size $(SUCCESS_PLOT_W)) --biases $(BIASES) --groups-count $(GROUPS_COUNT) --group-size $(SUCCESS_PLOT_GROUP_SIZE) --layout $(SUCCESS_PLOT_LAYOUT) --palette $(SUCCESS_PLOT_PALETTE) --output-format $(SUCCESS_PLOT_FORMAT) $(if $(filter 1,$(SUCCESS_PLOT_ENGLISH)),--english,)

K ?= 10
# NDCG tabulka: jeden vybraný bias — buď pořadí po seřazení β (0,1,2), nebo explicitní POPULATION_BIAS.
BIAS_INDEX ?= 0
# Prázdné ⇒ použije se BIAS_INDEX. Např. POPULATION_BIAS=1 nebo POPULATION_BIAS=2
POPULATION_BIAS ?=

table_ndcg_comparisions:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_ndcg_comparisions --k $(K) --window-size $(W) --groups-count $(GROUPS_COUNT) $(if $(strip $(POPULATION_BIAS)),--population-bias $(POPULATION_BIAS),--bias-index $(BIAS_INDEX))

table_unmatched_comparisions:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_unmatched_comparisions --window-size $(W) --bias 0

migrate-eval-cache-layout:
	python3 migrate_eval_cache_layout.py --window-size $(W) --eval-type test

table_rfc_large_group_size_comparisions:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_large_group_size_comparisions --window-size $(W) --groups-count $(GROUPS_COUNT) --group-sizes 3 5 7 10

LARGE_GROUP_SIZES ?= 5 7 10
ONLY_AVAILABLE ?= 1
table_rfc_large_group_size:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_large_group_size_comparisions --window-size $(W) --groups-count $(GROUPS_COUNT) --group-sizes $(LARGE_GROUP_SIZES) $(if $(filter 1,$(ONLY_AVAILABLE)),--only-available,)

# RFC vs group-size plot for large-group experiments (single W or tiled windows).
LARGE_RFC_PLOT_WINDOWS ?=
LARGE_RFC_PLOT_W ?= 1
LARGE_RFC_PLOT_LAYOUT ?= row
LARGE_RFC_PLOT_PALETTE ?= tab10
LARGE_RFC_PLOT_FORMAT ?= pdf
plot_rfc_large_group_size:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.plot_rfc_large_group_size $(if $(strip $(LARGE_RFC_PLOT_WINDOWS)),--windows $(LARGE_RFC_PLOT_WINDOWS),--window-size $(LARGE_RFC_PLOT_W)) --groups-count $(GROUPS_COUNT) --group-sizes $(LARGE_GROUP_SIZES) --layout $(LARGE_RFC_PLOT_LAYOUT) --palette $(LARGE_RFC_PLOT_PALETTE) --output-format $(LARGE_RFC_PLOT_FORMAT) $(if $(filter 1,$(ONLY_AVAILABLE)),--only-available,)

RFC_EXPORT_OUT ?= cache/cons_evaluations/exports/rfc_results_full.csv
export_rfc_full_csv:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.export_rfc_dataframe --all-runs --out-csv $(RFC_EXPORT_OUT)

RFC_SUCCESS_PREFIX ?= rfc_success
plot_rfc_window_and_success_group_size:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.plot_rfc_by_window_and_success_by_group_size --windows $(WINDOWS) --biases $(BIASES) --groups-count $(GROUPS_COUNT) --prefix $(RFC_SUCCESS_PREFIX) --all-runs

# =============================
# MOVIES / RESTAURANT (restored)
# =============================

restaurant-dataset-generate:
	python3 -m restaurant_data

restaurant-general-stats:
	python3 -m restaurant_data.general_analysis

restaurant-places-coverage-people:
	python3 -m restaurant_data.places_coverage.cumulative_coverage_with_people

restaurant-places-coverage:
	python3 -m restaurant_data.places_coverage.cumulative_coverage

restaurant-reviews-histogram:
	python3 -m restaurant_data.places_coverage.users_ratings_frequencies

restaurant-algo-comparison:
	python3 -m restaurant_data.algo_experiments.comparison_of_algorithms --min-user-rating $(CB_MIN_USER_RATING) --mode $(MODE)

restaurant-knn-by-test-size:
	python3 -m restaurant_data.algo_experiments.knn_by_dataset_size

restaurant-knn-neighbors-by-test-size:
	python3 -m restaurant_data.algo_experiments.knn_neighbors_count_by_test_size --mode $(MODE)

restaurant-knn-parameter-tuning:
	python3 -m restaurant_data.algo_experiments.knn_parameter_tuning

restaurant-sparsity-dataset:
	python3 -m restaurant_data.sparsity_issue.sparsity_users_elimination_method

restaurant-overlaps-by-test-size:
	python3 -m restaurant_data.sparsity_issue.overlaps_by_dataset_size

restaurant-cb-model-run:
	python3 -m restaurant_data.algo_experiments.algos.cb_model

CB_MIN_USER_RATING ?= 4
CB_K ?= 20
CB_TRAIN_RATIO ?= 0.8
restaurant-cb-evaluation:
	python3 -m restaurant_data.algo_experiments.cb_model_precision --min-user-rating $(CB_MIN_USER_RATING) --k $(CB_K) --train-ratio $(CB_TRAIN_RATIO) --mode $(MODE)

EASER_LAMBDAS ?= 100 200 400 800 1600 3200 6400
restaurant-optimal-easer-lambda:
	python3 -m restaurant_data.algo_experiments.optimal_easer_lambda --mode $(MODE) --regularization-options $(EASER_LAMBDAS)

movie-dataset-stats:
	python3 -m movies_data.genera_stats

movie-dataset-genre-frequencies:
	python3 -m movies_data.genres_coverage_frequencies

movie-popularity-histogram:
	python3 -m movies_data.popularity_histogram

init-size:
	python3 -m movies_data.initialisation_size

init-sampling-all:
	python3 -m movies_data.initialisation_sampling.init_sampling_test_all

movie-release-year-popularity:
	python3 -m movies_data.release_year_popularity_histogram

movie-knn-algo-comparison:
	python3 -m movies_data.algo_experiments.comparison_of_algorithms

movie-prod-filter:
	python3 -m movies_data.production_filtering.filter_dataset --input-dir ./dataset/movies/ml-32m/ --output-dir ./movies_data/production_filtering

movie-distribution-year-release-vs-easer:
	python3 -m movies_data.distribution_year_release_vs_easer

kmeans-analysis:
	python3 -m movies_data.initialisation_sampling.clustering.kmeans_analysis

