import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple
from lightfm import LightFM
import argparse
from evaluation_frameworks.consensus_evaluation.synthetic_groups.generator_tests import (
    validate_groups_min_interactions_run_wrapper,
    validate_divergent_groups_similarity_wrapper,
    validate_outlier_groups_similarity_wrapper,
    validate_similar_groups_similarity_wrapper,
)
import multiprocessing as mp
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from tqdm import tqdm
from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.model_train_load import train_or_load_lightfm_model
from evaluation_frameworks.consensus_evaluation.synthetic_groups.embeddings_extractor import EmbeddingExtractor
from evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_generator import GroupGenerator, GroupGeneratorRestrictedInteractions
from evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_testset_splitter import GroupsEvaluationSetsSplitter
from dataset.data_access import MovieLensDatasetLoader
from utils.config import load_or_build_pickle
from scipy.sparse import csr_matrix

# ====================================
# DESCRIPTION
# ====================================
# Preparation of evaluation datasets for the groups/consensus recommendation algorithms.
#
# Data are by default cached and stored into the cache directory in the root.
#
#
# Evaluation scripts rely on these data, thus it is good idea to run this script before performing evaluation.
#

#
# Configuration
# (mostly settings which data to load from cache/compute -- if the cache data are not found, we recompute everything)

LOAD_SPARSE_MATRIX = True
REBUILD_GROUPS = True

SKIP_VALIDATION = True # weather to skip correctness check of the generated groups

# ====================================
# HELPER FUNCTION
# ====================================

#region Helpers

def print_csr_stats(csr_matrix, name="Matrix"):
    num_users, num_items = csr_matrix.shape
    num_interactions = csr_matrix.nnz
    sparsity = (num_interactions / (num_users * num_items))
    avg_interactions_per_user = num_interactions / num_users
    avg_interactions_per_item = num_interactions / num_items

    print(f"=== {name} Statistics ===")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Interactions: {num_interactions}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Average Interactions per User: {avg_interactions_per_user:.2f}")
    print(f"Average Interactions per Item: {avg_interactions_per_item:.2f}")
    print()

def load_filtered_dataset(min_user_interactions, min_item_interactions, rating_threshold):
    preferred_dataset = "ml-32m"
    fallback_dataset = "ml-1m"

    dataset_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "dataset")

    def _dataset_has_core_files(dataset_name: str) -> bool:
        unified_root = os.path.join(dataset_root, "movies", dataset_name)
        movies_csv = os.path.join(unified_root, "movies.csv")
        ratings_csv = os.path.join(unified_root, "ratings.csv")
        return os.path.exists(movies_csv) and os.path.exists(ratings_csv)

    dataset_name = preferred_dataset
    if not _dataset_has_core_files(preferred_dataset):
        dataset_name = fallback_dataset
        print(
            f"⚠️ Dataset '{preferred_dataset}' is incomplete (missing movies.csv/ratings.csv). "
            f"Falling back to '{fallback_dataset}'."
        )

    data_loader = MovieLensDatasetLoader(dataset_name)
    movies_data_df, ratings_csr, user_id_map, movie_id_map = data_loader.load_sparse_ratings(LOAD_SPARSE_MATRIX)

    print("📊 Filtering original dataset... ")
    # filter only quality data at least 50 interactions per user and 20 interactions per item
    filtered_csr, filtered_user_map, filtered_item_map = data_loader.filter_csr_by_interaction_thresholds(
        ratings_csr,
        user_id_map,
        movie_id_map,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        rating_threshold=rating_threshold
    )

    ## Original movielens data
    print_csr_stats(ratings_csr, name="Original Ratings CSR")
    ## After filtering
    print_csr_stats(filtered_csr, name="Filtered Ratings CSR")

    return (movies_data_df, filtered_csr, filtered_user_map, filtered_item_map)

def load_eval_sets(group_type: str, groups: List[List[int]], group_count: int, min_common_items: int, val_count: int, test_count: int, *, group_size: int = 3, force_rebuild=False, unique = False) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    splitter = GroupsEvaluationSetsSplitter()
    unique_flag = "-unique" if unique else ""
    f_name = f"group-eval-{group_type}-{group_size}-eval_sets-filtered-ml-32m-{group_count}-min-common-{min_common_items}-val-{val_count}-test-{test_count}{unique_flag}.pkl"
    print("⚙️ Creation of group eval set, name:")
    print(f_name)
    return load_or_build_pickle(
        f_name,
        lambda: splitter.split_by_counts(groups, val_count, test_count),
        description=f"{group_type} groups evaluation set",
        force_rebuild=force_rebuild
    )

#endregion

def filter_disjoint_groups(groups: List[List[int]]) -> List[List[int]]:
    """
    Greedy filter: keep groups with no user overlap, deduplicate identical groups.
    - Input order determines which groups are kept.
    - Works for any group size (not just triplets).
    """
    selected: List[List[int]] = []
    used_users = set()
    seen_groups = set()  # dedup by membership, regardless of order

    for g in groups:
        g_set = frozenset(g)
        if g_set in seen_groups:
            continue  # skip exact duplicates (same members)
        if used_users.isdisjoint(g_set):
            selected.append(list(g))  # keep original order of members
            used_users.update(g_set)
        seen_groups.add(g_set)

    return selected

def generate_groups_size_3(filtered_csr, filtered_user_map, users_embeddings, groups_count: int = 10000, min_common_items: int = 10):

    groupsGen = GroupGeneratorRestrictedInteractions(filtered_csr, filtered_user_map, users_embeddings, 75, 25, 400)

    similar_groups_count = groups_count
    outlier_groups_count = groups_count
    random_groups_count = groups_count

    similar_groups_count = outlier_groups_count = random_groups_count = groups_count # nasty but ...

    similar_groups = load_or_build_pickle(
        f"groups-similar-filtered-ml-32m-{similar_groups_count}-min-common-{min_common_items}.pkl",
        lambda: groupsGen.generate_similar_group(groups_count=similar_groups_count, group_size=3, min_common_items=min_common_items),
        description="similar groups"
    )

    outlier_groups = load_or_build_pickle(
        f"groups-outlier-filtered-ml-32m-{outlier_groups_count}-min-common-{min_common_items}.pkl",
        lambda: groupsGen.generate_outlier_group_from_similar(groups_count=outlier_groups_count, similar_groups=similar_groups, min_common_items=min_common_items),
        description="outlier groups"
    )

    random_groups = load_or_build_pickle(
        f"groups-random-filtered-ml-32m-{random_groups_count}-min-common-{min_common_items}.pkl",
        lambda: groupsGen.generate_random_group(groups_count=random_groups_count, group_size=3, min_common_items=min_common_items),
        description="random groups"
    )

    divergent_groups_count = groups_count
    divergent_pickle_name = f"groups-divergent-filtered-ml-32m-{divergent_groups_count}-min-common-{min_common_items}.pkl"

    def _build_divergent_groups():
        return groupsGen.generate_divergent_group(
            groups_count=divergent_groups_count,
            group_size=3,
            min_common_items=min_common_items,
        )

    divergent_groups = load_or_build_pickle(
        divergent_pickle_name,
        _build_divergent_groups,
        description="divergent groups (all pairs sim <= to)",
    )
    if len(divergent_groups) == 0:
        print(
            "⚠️ Cached divergent groups list is empty (stale cache from old generator logic?). "
            "Rebuilding with force_rebuild=True…"
        )
        divergent_groups = load_or_build_pickle(
            divergent_pickle_name,
            _build_divergent_groups,
            description="divergent groups (all pairs sim <= to)",
            force_rebuild=True,
        )

    variance_groups_count = groups_count
    variance_groups = load_or_build_pickle(
        f"groups-variance-filtered-ml-32m-{variance_groups_count}-min-common-{min_common_items}.pkl",
        lambda: groupsGen.generate_variance_group(
            groups_count=variance_groups_count,
            group_size=3,
            min_common_items=min_common_items,
            min_avg_item_variance=1.2,
        ),
        description="variance groups (similar embeddings + high item-rating variance)",
    )

    print(f"Numbers of all groups:")
    print(f"🔗 similar {len(similar_groups)}")
    print(f"👽 outlier {len(outlier_groups)}")
    print(f"🎲 random {len(random_groups)}")
    print(f"🔻 divergent {len(divergent_groups)}")
    print(f"📈 variance {len(variance_groups)}")

    # disjoint (unique) per dataset
    similar_groups_unique = filter_disjoint_groups(similar_groups)
    outlier_groups_unique = filter_disjoint_groups(outlier_groups)
    random_groups_unique = filter_disjoint_groups(random_groups)
    divergent_groups_unique = filter_disjoint_groups(divergent_groups)
    variance_groups_unique = filter_disjoint_groups(variance_groups)

    print("Numbers of unique (disjoint) groups:")
    print(f"🔗 similar {len(similar_groups_unique)}")
    print(f"👽 outlier {len(outlier_groups_unique)}")
    print(f"🎲 random  {len(random_groups_unique)}")
    print(f"🔻 divergent {len(divergent_groups_unique)}")
    print(f"📈 variance {len(variance_groups_unique)}")

    # ===========================================
    # GENERATED GROUPS VALIDATION
    # ===========================================
    # We make sure obtained groups data meet required constrains.
    # That is: min items interactions in groups,
    #

    def validate_groups():

        print(f"📊 Validating generated groups: ")

        validate_outlier_groups_similarity_wrapper(
            groups=outlier_groups,
            user_embeddings=users_embeddings,
            ts=0.26,
            to=-0.12
        )

        validate_similar_groups_similarity_wrapper(
            groups=similar_groups,
            user_embeddings=users_embeddings,
            ts=0.26,
        )

        validate_groups_min_interactions_run_wrapper(
            groups=similar_groups,
            csr_matrix=filtered_csr,
            user_id_map=filtered_user_map,
            min_interactions=min_common_items
        )

        validate_groups_min_interactions_run_wrapper(
            groups=outlier_groups,
            csr_matrix=filtered_csr,
            user_id_map=filtered_user_map,
            min_interactions=min_common_items
        )

        validate_groups_min_interactions_run_wrapper(
            groups=random_groups,
            csr_matrix=filtered_csr,
            user_id_map=filtered_user_map,
            min_interactions=min_common_items
        )

        validate_divergent_groups_similarity_wrapper(
            groups=divergent_groups,
            user_embeddings=users_embeddings,
            to=-0.12,
        )

        validate_groups_min_interactions_run_wrapper(
            groups=divergent_groups,
            csr_matrix=filtered_csr,
            user_id_map=filtered_user_map,
            min_interactions=min_common_items,
        )

        # variance groups must still be similar in latent space
        validate_similar_groups_similarity_wrapper(
            groups=variance_groups,
            user_embeddings=users_embeddings,
            ts=0.26,
        )
        validate_groups_min_interactions_run_wrapper(
            groups=variance_groups,
            csr_matrix=filtered_csr,
            user_id_map=filtered_user_map,
            min_interactions=min_common_items,
        )

    if SKIP_VALIDATION:
        print("Skipping groups validation...")
    else:
        validate_groups()

    # ==============================================
    # GROUPS DATASET SPLIT
    # ==============================================
    # Split the groups data into train, validation a test sets.
    #

    groups_set_splitter = GroupsEvaluationSetsSplitter()

    val_groups_count = 1000
    test_groups_count = 1000

    similar_groups_eval_sets = load_eval_sets("similar", similar_groups, similar_groups_count, min_common_items, val_groups_count, test_groups_count)
    outlier_groups_eval_sets = load_eval_sets("outlier", outlier_groups, outlier_groups_count, min_common_items, val_groups_count, test_groups_count)
    random_groups_eval_sets = load_eval_sets("random", random_groups, random_groups_count, min_common_items, val_groups_count, test_groups_count)
    divergent_eval_sets = load_eval_sets(
        "divergent",
        divergent_groups,
        divergent_groups_count,
        min_common_items,
        val_groups_count,
        test_groups_count,
    )
    variance_eval_sets = load_eval_sets(
        "variance",
        variance_groups,
        variance_groups_count,
        min_common_items,
        val_groups_count,
        test_groups_count,
    )

    print(f"===================================")
    print(f"Numbers of all groups:")
    print(f"Similar:")
    print(f"    train {len(similar_groups_eval_sets[0])}")
    print(f"    validation {len(similar_groups_eval_sets[1])}")
    print(f"    test {len(similar_groups_eval_sets[2])}")
    print(f"Outlier:")
    print(f"    train {len(outlier_groups_eval_sets[0])}")
    print(f"    validation {len(outlier_groups_eval_sets[1])}")
    print(f"    test {len(outlier_groups_eval_sets[2])}")
    print(f"Random:")
    print(f"    train {len(random_groups_eval_sets[0])}")
    print(f"    validation {len(random_groups_eval_sets[1])}")
    print(f"    test {len(random_groups_eval_sets[2])}")
    print(f"Divergent:")
    print(f"    train {len(divergent_eval_sets[0])}")
    print(f"    validation {len(divergent_eval_sets[1])}")
    print(f"    test {len(divergent_eval_sets[2])}")
    print(f"Variance:")
    print(f"    train {len(variance_eval_sets[0])}")
    print(f"    validation {len(variance_eval_sets[1])}")
    print(f"    test {len(variance_eval_sets[2])}")


def generate_groups_size_any(filtered_csr, filtered_user_map, users_embeddings, group_size: int, groups_count: int = 10000, min_common_items: int = 10):

    groupsGen = GroupGeneratorRestrictedInteractions(filtered_csr, filtered_user_map, users_embeddings, 75, 25, 400)

    random_groups_count = groups_count

    random_groups = load_or_build_pickle(
        f"groups-random-filtered-ml-32m-{random_groups_count}-min-common-{min_common_items}-size-{group_size}.pkl",
        lambda: groupsGen.generate_random_group(groups_count=random_groups_count, group_size=group_size, min_common_items=min_common_items),
        description="random groups"
    )

    print(f"🎲 random {len(random_groups)} of size {group_size}")

    random_groups_unique = filter_disjoint_groups(random_groups)

    print(f"🎲 unique random {len(random_groups)} of size {group_size}")

    val_groups_count = 1000
    test_groups_count = 1000

    random_groups_eval_sets = load_eval_sets("random", random_groups, random_groups_count, min_common_items, val_groups_count, test_groups_count, group_size=group_size)

    print(f"===================================")
    print(f"Numbers of all groups:")
    print(f"Random:")
    print(f"    train {len(random_groups_eval_sets[0])}")
    print(f"    validation {len(random_groups_eval_sets[1])}")
    print(f"    test {len(random_groups_eval_sets[2])}")


    return random_groups






if __name__== "__main__":

    # ======================================
    # Dataset load and preparation
    # ======================================

    print("Loading data as CSR... ")

    #
    # load MovieLens 32m -- that is 32m interactions
    # For efficiency we are using CSR -- Compressed sparse format
    #

    parser = argparse.ArgumentParser(description="Synthetic groups data preparation script.")
    parser.add_argument("--group-size", type=int, default=3, help="Size of groups to generate. (similar, outlier, divergent, variance only for group size 3). " \
        "Larger value generates")
    parser.add_argument("--min-com", type=int, default=10, help="Minimal number of common items in the group " \
        "Larger value generates")
    args = parser.parse_args()



    group_size = args.group_size


    groups_count = 100000
    min_common_items_in_group = args.min_com # items all users rated

    min_user_interactions = 50
    min_item_interactions = 20
    rating_threshold = 4

    movies_data_df, filtered_csr, filtered_user_map, filtered_item_map = load_filtered_dataset(min_user_interactions, min_item_interactions, rating_threshold)

    # =================================================
    # CREATE USER EMBEDDINGS
    # =================================================
    #
    # To find similar groups we need user embeddings. To this end we use LightFM model.
    #

    model_cache_name = f"lightfm-ml-32m-filtered-interactions-min-{min_user_interactions}-users-{min_item_interactions}-{rating_threshold}-items"
    model = train_or_load_lightfm_model(filtered_csr, model_name=model_cache_name)

    embeddingExtract = EmbeddingExtractor(model_type="lightfm")

    users_embeddings = load_or_build_pickle(f"group-algorithm-eval-users-embeddings-{model_cache_name}.pkl",
                        lambda: embeddingExtract.extract_user_embeddings(model, filtered_user_map),
                        description="user embeddings LightFm for group algorithm eval"
                    )

    # =================================================
    # GENERATE GROUPS
    # =================================================
    #
    # To find similar groups we need user embeddings. To this end we use LightFM model.
    #

    #
    # Generate groups:
    # We generate 5 types of groups of size 3.
    # 1. Similar --> all members are similar using cosine similarity (they are in 75th percentile)
    # 2. Outlier --> two members are similar (similarity is in the 75th percentile) and the third is distant from both (he is in 25th percentile from both)
    # 3. Random --> members are selected randomly
    # 4. Divergent --> all three pairs have similarity <= `to` (same low threshold as outlier dissimilarity)
    # 5. Variance --> similar in latent space, but high rating variance on common items
    #
    # Overlaps of size <= 2 are allowed, otherwise groups are unique.
    #

    if (group_size == 3):
        generate_groups_size_3(filtered_csr, filtered_user_map, users_embeddings, groups_count=groups_count, min_common_items=min_common_items_in_group)
    else:
        generate_groups_size_any(filtered_csr, filtered_user_map, users_embeddings, group_size, groups_count=groups_count, min_common_items=min_common_items_in_group)




