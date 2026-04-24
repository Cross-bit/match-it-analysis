from collections import defaultdict
from time import time
from typing import DefaultDict, Dict, List, Optional, Set, Tuple
from lightfm import LightFM

from evaluation_frameworks.consensus_evaluation.synthetic_groups.generator_tests import validate_groups_min_interactions_run_wrapper, validate_outlier_groups_similarity_wrapper, validate_similar_groups_similarity_wrapper
import multiprocessing as mp
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from tqdm import tqdm
from evaluation_frameworks.consensus_evaluation.synthetic_groups.embeddings_extractor import EmbeddingExtractor
from evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_generator import GroupGenerator, GroupGeneratorRestrictedInteractions
from evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_testset_splitter import GroupsEvaluationSetsSplitter
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserSparse
from dataset.data_access import MovieLensDatasetLoader
from utils.config import load_or_build_pickle
from scipy.sparse import csr_matrix


def train_or_load_lightfm_model(csr_data, *, model_name, dims=50, epochs=10, loss="warp"):
    """ Trains or load """

    def train_model():
        model = LightFM(no_components=dims, loss=loss, random_state=42)
        model.fit(csr_data, epochs=epochs, num_threads=4)
        return model

    model = load_or_build_pickle(
        model_name,
        train_model,
        description=f"LightFM model ({loss}, {dims} dims, {epochs} epochs)"
    )
    return model


def train_or_load_easer_model(csr_data, user_id_map, item_id_map, *, model_name, regularization=10000, force_rebuild=False):
    """ Trains or load """

    def train_model():
        ts = time()
        model = EaserSparse(regularization)
        model.fit(csr_data, user_id_map, item_id_map)
        te = time()
        print(f"⏱️ Train time: {te-ts:.2f} s")

        return model

    model = load_or_build_pickle(
        model_name,
        train_model,
        description=f"Easer model ({regularization} regularization)",
        force_rebuild=force_rebuild
    )

    return model
