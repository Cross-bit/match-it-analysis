from collections import Counter
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


if __name__ == "__main__":

    t_param = 7
    rec_window_size = 10
    group_sample_size = 1000

    r = Runner("train", "similar")

    for w in range(5, 20, 5):
        for t in range(int(w * 0.2), int(w * 0.9), int(w * 0.2)):

            factory_method = lambda single_user_model, evaluation_set_csr: (
                AsyncMediatorFactoryBuilder()
                .with_recommender_engine(lambda group: RecommendationEngineIndividualEaser(group, model_iterator=single_user_model))
                #.with_recommender(lambda group: RecommendationEngineGroupAllIndividualEaser(group, evaluation_set_csr, model_iterator=GR_AggregatedRecommendations(single_user_model)))
                .with_priority_function(lambda group: SimplePriorityFunction(group, algorithm=single_user_model))
                .with_threshold_policy(lambda: ThresholdPolicyStatic(t_param=t_param))
                .with_redistribution(lambda group, pf: RedistributionUnit(group, pf))
                .with_mediator(lambda group, rec, ru, th: ConsensusMediatorAsyncApproach(group, rec, ru, th, window_size=rec_window_size))
                .build()
            )

            print(f"{w}, {t}")
            print(f"Static for t_param {t}")
            res = r.run(factory_method, group_sample_size, window_size=rec_window_size)
            print_evaluation_result(res)


