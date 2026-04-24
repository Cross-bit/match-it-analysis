"""
Build **evaluation contexts** (sparse matrices, group lists, ground truth) and **vote simulators**.

``build_context*`` / ``build_context_holdout`` wire dataset pickles from ``evaluation_preparation`` into
the dict that ``Runner.refresh_context`` expects. Agent loaders construct ``UserVoteSimulator`` variants
with trained EASer/LightFM models for the evaluation split.
"""

from collections import Counter
from typing import Dict, List, Literal, Set, Tuple
from lightfm import LightFM
from time import time

import numpy as np
from tqdm import tqdm
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_evaluator import ConsensusAgentBasedEvaluator
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import AsyncMediatorFactoryBuilder, SyncMediatorFactoryBuilderSync
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_data_interpreter import print_evaluation_result
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import GeneralRecommendationEngineBase, RecommendationEngineGroupAllIndividualEaserUpdatable, RecommendationEngineGroupAllSameEaser, RecommendationEngineIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import PriorityFunction, RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ConsensusMediatorBase, ConsensusMediatorSyncApproach, ThresholdPolicy, ThresholdPolicyStatic
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_evaluation_agents.evaluation_agent import UserVoteSimulator, UserVoteSimulatorSigmoidNormed
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import GR_AggregatedRecommendations
from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation import filter_disjoint_groups, load_eval_sets, load_filtered_dataset
from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.ground_truth_filtering import prepare_group_eval_data, prepare_group_eval_data2_test_split
from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.model_train_load import train_or_load_easer_model, train_or_load_lightfm_model
from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoBase
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserSparse
from evaluation_frameworks.general_recommender_evaluation.algorithms.svd import SurpriseSVDRecommender
from utils.config import load_from_pickle, load_or_build_pickle
from scipy.sparse import csr_matrix
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.debug_profile import log_event, timed


#
#
# CONTEXT LOADS
#
# ----------------------------------------------------------------------------
#


def build_context_groups_data(EVALUATION_SET_TYPE=Literal["train", "validation", "test"], GROUP_TYPE=Literal["similar", "outlier", "random", "divergent", "variance"], unique_groups = False, in_group_min_common_items = 10):
    """Retrieves data for groups evaluation.

    unique_groups -- groups must be unique.

    Returns:
        _type_: _description_
    """

    # Load only the requested group-type pickle (same on-disk data as before; avoids loading all 5 lists per refresh).
    groups_path = f"groups-{GROUP_TYPE}-filtered-ml-32m-100000-min-common-{in_group_min_common_items}.pkl"
    GROUPS_DATA_ALL: List[List[int]] = load_from_pickle(groups_path)
    print(len(GROUPS_DATA_ALL))

    if unique_groups:
        GROUPS_DATA_ALL = filter_disjoint_groups(GROUPS_DATA_ALL)

    VALIDATION_SIZE = 1000
    TEST_SIZE = 1000

    # Select the evaluation set
    TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS = load_eval_sets(
        GROUP_TYPE,
        GROUPS_DATA_ALL,
        100000,
        in_group_min_common_items,
        VALIDATION_SIZE,
        TEST_SIZE,
        force_rebuild=False,
        unique=unique_groups,
    )
    EVAL_SET_GROUPS_DATA: List[List[int]] = []

    if (EVALUATION_SET_TYPE == "train"):
        EVAL_SET_GROUPS_DATA = TRAIN_GROUPS
    elif (EVALUATION_SET_TYPE == "validation"):
        EVAL_SET_GROUPS_DATA = VAL_GROUPS
    elif (EVALUATION_SET_TYPE == "test"):
        EVAL_SET_GROUPS_DATA = TEST_GROUPS

    return EVAL_SET_GROUPS_DATA, GROUPS_DATA_ALL

def build_context_large_groups_data(EVALUATION_SET_TYPE=Literal["train", "validation", "test"], GROUP_TYPE=Literal["random"], group_size = 10, unique_groups = False, in_group_min_common_items = 10):
    """Retrieves data for groups evaluation.

    unique_groups -- groups must be unique.

    Returns:
        _type_: _description_
    """

    #similar_groups:List[List[int]] = load_from_pickle(f"groups-similar-filtered-ml-32m-100000-min-common-{in_group_min_common_items}.pkl")
    #outlier_groups:List[List[int]] = load_from_pickle(f"groups-outlier-filtered-ml-32m-100000-min-common-{in_group_min_common_items}.pkl")
    try:
        random_groups: List[List[int]] = load_from_pickle(
            f"groups-random-filtered-ml-32m-100000-min-common-{in_group_min_common_items}-size-{group_size}.pkl"
        )
    except FileNotFoundError:
        if group_size == 3:
            # Backward compatibility: historical size-3 random groups were stored without "-size-3".
            random_groups = load_from_pickle(
                f"groups-random-filtered-ml-32m-100000-min-common-{in_group_min_common_items}.pkl"
            )
        else:
            raise

    if (unique_groups): # quick operation, no need to load/store, just makes sure all the users are unique across all the groups
        random_groups = filter_disjoint_groups(random_groups)

    VALIDATION_SIZE = 1000
    TEST_SIZE = 1000

    GROUPS_DATA_ALL = []
    GROUPS_DATA_ALL = random_groups

    # Select the evaluation set
    TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS = load_eval_sets(
        GROUP_TYPE,
        GROUPS_DATA_ALL,
        100000,
        in_group_min_common_items,
        VALIDATION_SIZE,
        TEST_SIZE,
        group_size=group_size,
        force_rebuild=False,
        unique=unique_groups,
    )
    EVAL_SET_GROUPS_DATA: List[List[int]] = []

    if (EVALUATION_SET_TYPE == "train"):
        EVAL_SET_GROUPS_DATA = TRAIN_GROUPS
    elif (EVALUATION_SET_TYPE == "validation"):
        EVAL_SET_GROUPS_DATA = VAL_GROUPS
    elif (EVALUATION_SET_TYPE == "test"):
        EVAL_SET_GROUPS_DATA = TEST_GROUPS

    return EVAL_SET_GROUPS_DATA, GROUPS_DATA_ALL



def build_context(EVALUATION_SET_TYPE: Literal["train", "validation", "test"], GROUP_TYPE: Literal["similar", "outlier", "random", "divergent", "variance"], unique_groups = False):
    """Retrieves data for groups evaluation.

    unique_groups -- groups must be unique.

    Returns:
        _type_: _description_
    """

    min_user_interactions = 50
    min_item_interactions = 20
    rating_threshold = 4
    min_common_items = 10

    _, ratings_csr, user_id_map, item_id_map = load_filtered_dataset(
        min_user_interactions, min_item_interactions, rating_threshold
    )

    EVAL_SET_GROUPS_DATA, GROUPS_DATA_ALL = build_context_groups_data(EVALUATION_SET_TYPE, GROUP_TYPE, unique_groups, in_group_min_common_items=min_common_items)

    # ==================================
    # EVALUATION PHASE 1. -- GROUPS EVALUATION
    # ==================================
    # In this phase we evaluate group metrics.
    #
    #
    #

    # groups_ground_truth - group to the common ground truth items
    # filtered_evaluation_set_csr - matrix without the common ground-truth items
    filtered_evaluation_set_csr, groups_ground_truth = load_or_build_pickle(
        f"{EVALUATION_SET_TYPE}-groups-evaluation-{len(EVAL_SET_GROUPS_DATA)}.pkl",
        lambda: prepare_group_eval_data(EVAL_SET_GROUPS_DATA, ratings_csr, user_id_map, item_id_map),
        description="filtering ground-truth items"
    )

    return {
        "ratings_csr": ratings_csr,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "filtered_evaluation_set_csr": filtered_evaluation_set_csr,
        "groups_ground_truth": groups_ground_truth,
        "eval_set_groups_data": EVAL_SET_GROUPS_DATA,
        "groups_data_all": GROUPS_DATA_ALL
    }

def build_context_holdout(EVALUATION_SET_TYPE=Literal["train", "validation", "test"], GROUP_TYPE=Literal["similar", "outlier", "random", "divergent", "variance"], unique_groups = False, test_ratio = 0.5, force_rebuild = False):
    """Retrieves data for groups evaluation.

    Args:
        unique_groups -- groups must be unique.
        force_rebuild --

    Returns:
        _type_: _description_
    """

    min_user_interactions = 50
    min_item_interactions = 20
    rating_threshold = 4
    min_common_items = 10

    with timed("context.dataset.load_filtered"):
        _, ratings_csr, user_id_map, item_id_map = load_filtered_dataset(
            min_user_interactions, min_item_interactions, rating_threshold
        )

    with timed("context.groups.load", extra={"eval_type": EVALUATION_SET_TYPE, "group_type": GROUP_TYPE}):
        EVAL_SET_GROUPS_DATA, GROUPS_DATA_ALL = build_context_groups_data(EVALUATION_SET_TYPE, GROUP_TYPE, unique_groups, in_group_min_common_items=min_common_items)

    filtered_evaluation_set_csr: csr_matrix
    users_ground_truth: Dict[int, Set[int]]
    groups_ground_truth: Dict[Tuple[int, ...], Set[int]]

    # groups_ground_truth - group to the common ground truth items
    # filtered_evaluation_set_csr - matrix without the common ground-truth items
    with timed("context.prepare_holdout", extra={"groups_input": len(EVAL_SET_GROUPS_DATA)}):
        filtered_evaluation_set_csr, kept_evaluation_groups, users_ground_truth, groups_ground_truth = load_or_build_pickle(
            f"{EVALUATION_SET_TYPE}-groups-evaluation-testsplit-{min_common_items}-{len(EVAL_SET_GROUPS_DATA)}-hold-out.pkl",
            lambda: prepare_group_eval_data2_test_split(EVAL_SET_GROUPS_DATA, ratings_csr, user_id_map, item_id_map, test_ratio=test_ratio),
            description="filtering ground-truth items"
        )

    print("test sets data: ")
    print(len(users_ground_truth))
    print(len(groups_ground_truth))
    print(filtered_evaluation_set_csr.shape)
    log_event(
        "context.holdout.ready",
        extra={
            "users_ground_truth": len(users_ground_truth),
            "groups_ground_truth": len(groups_ground_truth),
            "kept_groups": len(kept_evaluation_groups),
            "shape": list(filtered_evaluation_set_csr.shape),
        },
    )

    return {
        "ratings_csr": ratings_csr,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "filtered_evaluation_set_csr": filtered_evaluation_set_csr,
        "groups_ground_truth": groups_ground_truth, # GT items for the
        "users_ground_truth": users_ground_truth, # GT items for individual users (the groups_ground_truth is subset of these items)
        "eval_set_groups_data": kept_evaluation_groups, # IT IS IMPORTANT TO USE ONLY VALID TEST SET GROUPS HERE!!!!
        "groups_data_all": GROUPS_DATA_ALL
    }

def build_context_large_holdout(EVALUATION_SET_TYPE=Literal["train", "validation", "test"], GROUP_TYPE=Literal["random"], group_size=10, unique_groups = False, test_ratio = 0.5, force_rebuild = False):
    """Retrieves data for large groups evaluation.

    Args:
        unique_groups -- groups must be unique.
        force_rebuild --

    Returns:
        _type_: _description_
    """

    min_user_interactions = 50
    min_item_interactions = 20
    rating_threshold = 4
    min_common_items = 10

    with timed("context_large.dataset.load_filtered"):
        _, ratings_csr, user_id_map, item_id_map = load_filtered_dataset(
            min_user_interactions, min_item_interactions, rating_threshold
        )

    with timed("context_large.groups.load", extra={"eval_type": EVALUATION_SET_TYPE, "group_size": group_size}):
        EVAL_SET_GROUPS_DATA, GROUPS_DATA_ALL = build_context_large_groups_data(EVALUATION_SET_TYPE, GROUP_TYPE, group_size, unique_groups, in_group_min_common_items=min_common_items)


    filtered_evaluation_set_csr: csr_matrix
    users_ground_truth: Dict[int, Set[int]]
    groups_ground_truth: Dict[Tuple[int, ...], Set[int]]

    # groups_ground_truth - group to the common ground truth items
    # filtered_evaluation_set_csr - matrix without the common ground-truth items
    with timed("context_large.prepare_holdout", extra={"groups_input": len(EVAL_SET_GROUPS_DATA), "group_size": group_size}):
        filtered_evaluation_set_csr, kept_evaluation_groups, users_ground_truth, groups_ground_truth = load_or_build_pickle(
            f"{EVALUATION_SET_TYPE}-groups-evaluation-testsplit-{min_common_items}-size-{group_size}-{len(EVAL_SET_GROUPS_DATA)}-hold-out.pkl",
            lambda: prepare_group_eval_data2_test_split(EVAL_SET_GROUPS_DATA, ratings_csr, user_id_map, item_id_map, test_ratio=test_ratio),
            description="filtering ground-truth items"
        )

    print("test sets data: ")
    print(len(users_ground_truth))
    print(len(groups_ground_truth))
    print(filtered_evaluation_set_csr.shape)
    log_event(
        "context_large.holdout.ready",
        extra={
            "users_ground_truth": len(users_ground_truth),
            "groups_ground_truth": len(groups_ground_truth),
            "kept_groups": len(kept_evaluation_groups),
            "shape": list(filtered_evaluation_set_csr.shape),
            "group_size": group_size,
        },
    )

    return {
        "ratings_csr": ratings_csr,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "filtered_evaluation_set_csr": filtered_evaluation_set_csr,
        "groups_ground_truth": groups_ground_truth, # GT items for the
        "users_ground_truth": users_ground_truth, # GT items for individual users (the groups_ground_truth is subset of these items)
        "eval_set_groups_data": kept_evaluation_groups, # IT IS IMPORTANT TO USE ONLY VALID TEST SET GROUPS HERE!!!!
        "groups_data_all": GROUPS_DATA_ALL
    }

#
#
# EVALUATION AGENTS LOADS
#
# ----------------------------------------------------------------------------
#

def load_evaluation_agent_base(filtered_evaluation_set_csr: csr_matrix, user_id_map: Dict[int, int], item_id_map: Dict[int, int], evaluation_set_type: str, identifier="") -> UserVoteSimulator:
    simulation_model_name = f"{evaluation_set_type}-groups-evaluation-agent-svd{identifier}.pkl"
    model: RecAlgoBase = load_or_build_pickle(
            simulation_model_name,
            lambda: SurpriseSVDRecommender(embeddings_dims=50).fit(filtered_evaluation_set_csr, user_id_map, item_id_map),
            description=f"SVD model (embeddings dim {50})")

    # agent
    simulation_agent = UserVoteSimulator(model, filtered_evaluation_set_csr, user_id_map, item_id_map)

    return simulation_agent


def load_evaluation_agent_sigmoid_normed(filtered_evaluation_set_csr: csr_matrix,
                                        user_id_map: Dict[int, int],
                                        item_id_map: Dict[int, int],
                                        evaluation_set_type: str,
                                        rating_threshold = 3.5, # prediction >3.5 => like
                                        tau_norm: float = 0.7,  # skewness of the sigmoid (how hard the decisions are)
                                        delta_raw: float = 1.0,  # the width of the neutral range (e.g. for rating_threshold = 3.5, we have 2.5< => dislike; (2.5, 3.5) => neutral; >3.5 => like)
                                        global_user_bias: float = 0.0, # the global pessimist/optimist bias (whether users should vote more optimistically or pessimistically)
                                        normalization_sample_k: int = 500, # number of samples to use to estimate μ,σ to compute z-score
                                        use_per_user_biases = False,
                                        seed = 42,
                                        identifier="",
                                        force_rebuild_cache=False
                                        ) -> UserVoteSimulatorSigmoidNormed:
    """ Loads evaluation context

    Args:
        filtered_evaluation_set_csr (csr_matrix): _description_
        user_id_map (Dict[int, int]): _description_
        item_id_map (Dict[int, int]): _description_
        evaluation_set_type (str): _description_
        rating_threshold (float, optional): _description_. Defaults to 3.5.
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        UserVoteSimulatorSigmoidNormed: _description_
    """

    simulation_model_name = f"{evaluation_set_type}-groups-evaluation-agent-sigmoid-normed-svd{identifier}.pkl"
    model: RecAlgoBase = load_or_build_pickle(
            simulation_model_name,
            lambda: SurpriseSVDRecommender(embeddings_dims=50).fit(filtered_evaluation_set_csr, user_id_map, item_id_map),
            description=f"SVD model (embeddings dim {50})",
            force_rebuild=force_rebuild_cache
            )

    # agent
    simulation_agent = UserVoteSimulatorSigmoidNormed(
                            model,
                            filtered_evaluation_set_csr,
                            user_id_map,
                            item_id_map,
                            global_bias=global_user_bias,
                            rating_threshold=rating_threshold,
                            tau_norm=tau_norm,
                            delta_raw=delta_raw,
                            normalization_sample_k=normalization_sample_k,
                            use_per_user_biases=use_per_user_biases,
                            seed=seed
                        )

    return simulation_agent