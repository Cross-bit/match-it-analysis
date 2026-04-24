from collections import Counter
import os
from typing import Dict, List, Literal, Set, Tuple
from lightfm import LightFM
from time import time

import numpy as np
from tqdm import tqdm
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluation_context_factory import build_context, build_context_holdout, load_evaluation_agent_base, load_evaluation_agent_sigmoid_normed
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_runner import Runner
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_evaluator import ConsensusAgentBasedEvaluator
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.consensus_mediator_factories import AsyncMediatorFactoryBuilder
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluators.evaluation_data_interpreter import print_evaluation_result
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine import RecommendationEngineGroupAllIndividualEaserUpdatable, RecommendationEngineIndividualEaser
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit import RedistributionUnit, SimplePriorityFunction
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorAsyncApproach, ThresholdPolicyStatic
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import GR_AggregatedRecommendations
from evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.model_train_load import train_or_load_easer_model
from evaluation_frameworks.general_recommender_evaluation.algorithms.easer import EaserSparse
from utils.config import load_from_pickle, load_or_build_pickle
from scipy.sparse import csr_matrix

EVALUATION_NAME=os.path.basename(__file__)

output_name = "async_static_threshold_grid_search"

group_sample_size = 100

r = Runner("train", "similar")

results = {}

for w in range(5, 20, 5):
    for t in range(int(w * 0.2), int(w * 0.9), int(w * 0.2)):

        # Pozor: uvnitř používáme t a w z aktuální iterace
        factory_method = lambda single_user_model, evaluation_set_csr, t=t, w=w: (
            AsyncMediatorFactoryBuilder()
            .with_recommender_engine(
                lambda group: RecommendationEngineIndividualEaser(group, model_iterator=single_user_model))
            .with_priority_function(
                lambda group, model=single_user_model:
                    SimplePriorityFunction(group, algorithm=model)
            )
            .with_threshold_policy(lambda t_=t: ThresholdPolicyStatic(t_param=t_))
            .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
            .with_mediator(
                lambda group, rec, ru, th, w_=w:
                    ConsensusMediatorAsyncApproach(group, rec, ru, th, window_size=w_)
            )
            .build()
        )

        res = r.run(factory_method, group_sample_size, window_size=w)
        results[(w, t)] = res

# --- final print ---
for (w, t), res in sorted(results.items()):
    print(f"{w}, {t}")
    print(f"Static for t_param {t}")
    print_evaluation_result(res)