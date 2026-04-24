"""
Simulated **users** during evaluation: ``UserVoteSimulator`` scores mediator-proposed items with a
trained single-user recommender, compares scores to a like threshold, and feeds ``Vote`` objects back
into the mediator for the next round.

This is not a human study — it is the "evaluation agent" that makes async/sync/hybrid mediators
comparable under identical preference noise assumptions.
"""

from collections import defaultdict
from time import perf_counter
from typing import List, Literal, Optional, Dict, Tuple
import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix

from evaluation_frameworks.consensus_evaluation.consensus_algorithm.models import Vote
from evaluation_frameworks.consensus_evaluation.consensus_mediator import ConsensusMediatorBase
from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoBase
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.debug_profile import (
    is_enabled,
    sim_add_time,
    sim_incr,
)

class UserVoteSimulator():
    """
    Simulates group decision-making using user preference predictions.
    """

    def __init__(
        self,
        model: RecAlgoBase,
        interaction_matrix: csr_matrix,
        user_id_map,
        item_id_map,
        rating_threshold: float = 3.5,
    ) -> None:
        """
        Args:
            model (RecAlgoBase): Trained single-user recommendation model.
            interaction_matrix (csr_matrix): Sparse user-item interaction matrix.
            rating_threshold (float): Score threshold above which a user 'likes' the item.
        """
        self.rec_algo: RecAlgoBase = model

        self.rating_threshold = rating_threshold
        self.num_users, self.num_items = interaction_matrix.shape

        self._user_id_to_index = {external_id: idx for idx, external_id in user_id_map.items()}
        self._item_id_to_index = {external_id: idx for idx, external_id in item_id_map.items()}

    def get_metadata(self):
        return {
            "type": self.__class__.__name__,
            "score_prediction_model_type": self.rec_algo.__class__.__name__,
            "rating_threshold": self.rating_threshold
        }

    def simulate_group_decision(
        self,
        group: List[int],
        mediator: ConsensusMediatorBase,
        end_on_first_match: bool,
        max_rounds: int = 10,
    ) -> Dict[str, Optional[object]]:
        """
        Simulate iterative group recommendation process with voting and consensus.

        Args:
            group (List[int]): List of user IDs in the group.
            mediator: Consensus recommender that provides per-user recommendations per round.
            max_rounds (int): Max number of rounds before giving up.
            end_on_first_match (int): If True, yields list of size one of the first matched item.

        Returns:
            dict with:
                - "round_found": int or None
                - "agreed_item": item ID or None
                - "all_votes": List of round vote summaries
        """

        all_votes: List[Dict[int, List[str]]] = []
        last_round_votes: Dict[int, List[Vote]] = {user_id: [] for user_id in group}

        agreed_items: List[int] = []
        agreed_items_round_found: List[int] = []
        all_recommendations: List[Dict[int, List[int]]] = []
        _prof = is_enabled()
        for round_i in range(max_rounds + 1):
            if _prof:
                t_chk = perf_counter()
            agreed_item_ids = mediator.check_matches(last_round_votes)

            for agreed_item_id in agreed_item_ids:
                agreed_items.append(agreed_item_id)
                agreed_items_round_found.append(round_i)

                if end_on_first_match:
                    if _prof:
                        sim_add_time("sim.round.check_and_clear", perf_counter() - t_chk)
                        sim_incr("sim.rounds", 1)
                    return {
                        "round_found": [round_i],
                        "agreed_item": [agreed_item_id],
                        "all_recommendations": all_recommendations,
                        "all_votes": all_votes
                    }
                else:
                    mediator.clear_item_votes(agreed_item_id) # we reset matched item votes so the matching can continue

            if _prof:
                sim_add_time("sim.round.check_and_clear", perf_counter() - t_chk)

            if _prof:
                t_rec = perf_counter()
            recommendations = mediator.get_next_round_recommendation(last_round_votes)
            if _prof:
                sim_add_time("sim.round.get_recommendations", perf_counter() - t_rec)

            all_recommendations.append(recommendations)

            round_votes: Dict[int, List[str]] = {}
            last_round_votes = {}

            if _prof:
                t_vote = perf_counter()
            vote_predict_calls = 0
            # Every user votes over his recommended items
            for user_id in group:
                voted_items = []
                recommended_items = recommendations[user_id]
                vote_predict_calls += len(recommended_items)

                for item_id in recommended_items:

                    vote_value = self.simulate_user_vote(user_id, item_id)

                    voted_items.append(Vote(item_id, vote_value))
                    round_votes.setdefault(item_id, []).append(vote_value)

                last_round_votes[user_id] = voted_items

            if _prof:
                sim_add_time("sim.round.voting", perf_counter() - t_vote)
                sim_incr("sim.vote_calls", vote_predict_calls)
                sim_incr("sim.rounds", 1)

            all_votes.append(round_votes)

        return {
            "round_found": agreed_items_round_found,
            "agreed_item": agreed_items,
            "all_recommendations": all_recommendations,
            "all_votes": all_votes
        }

    def simulate_user_vote(self, user_id: int, item_id: int) -> int:
        # 1. get prediction
        score = self.rec_algo.predict(user_id, item_id)

        # 2. set thresholds
        t0 = self.rating_threshold - 1.0   # e.g. 2.5, for threshold 3.5
        t1 = self.rating_threshold         # e.g. 3.5
        tau = 0.5                          # smaller => sharper decision making

        # 2. find probabilities
        p_like = self._sigmoid((score - t1) / tau)      # chance for like
        p_dislike = self._sigmoid((t0 - score) / tau)   # chance for dislike
        p_neutral = 1.0 - p_like - p_dislike       # rest is neutral

        # 3. normalize
        p = np.clip([p_dislike, p_neutral, p_like], 0.0, 1.0)
        s = p.sum()

        if s == 0:
            p = np.array([0.4, 0.2, 0.4])  # fallback
        else:
            p = p / s

        # 4. sample
        return np.random.choice([-1, 0, 1], p=p)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict_user_scores(self, user_id: int) -> np.ndarray:
        """
        Predict scores for all items for a given user.
        """
        return self.rec_algo.predict(user_id, np.arange(self.num_items))



class UserVoteSimulatorSigmoidNormed(UserVoteSimulator):

    def __init__(self,
                model: RecAlgoBase,
                interaction_matrix: csr_matrix,
                user_id_map: Dict[int, int],
                item_id_map: Dict[int, int],
                rating_threshold = 3.5, # prediction >3.5 => like
                tau_norm: float = 0.7,  # skewness of the sigmoid (how hard the decisions are)
                delta_raw: float = 1.0,  # the width of the neutral range (e.g. for rating_threshold = 3.5, we have 2.5< => dislike; (2.5, 3.5) => neutral; >3.5 => like)
                global_bias: float = 0.0, # the global pessimist/optimist bias (whether users should vote more optimistically or pessimistically)
                normalization_sample_k: int = 500, # number of samples to use to estimate μ,σ to compute z-score
                use_per_user_biases = True,
                seed = 42
                ):

        super().__init__(model, interaction_matrix, user_id_map, item_id_map, rating_threshold)

        self._seed = seed
        self.rng = np.random.default_rng(self._seed)

        self.tau_norm = float(tau_norm)
        self.delta_raw = float(delta_raw)
        self.global_bias = float(global_bias)
        self.sample_k = normalization_sample_k
        self.min_sigma = 0.5 # min σ for small std
        self._user_scale: Dict[int, tuple] = {}

    def get_metadata(self):
        base = super().get_metadata()
        base["tau_norm"] = self.tau_norm
        base["delta_raw"] = self.delta_raw
        base["global_bias"] = self.global_bias
        base["normalization_sample_k"] = self.sample_k
        base["min_normalization_sigma"] = self.min_sigma
        base["sampling_seed"] = self._seed
        return base

    def _get_user_scale(self, user_id: int) -> tuple[float, float]:
        """Return (mu_u, sigma_u) predicted scores for given user"""

        if user_id in self._user_scale: # use cache
            if is_enabled():
                sim_incr("sim.user_scale.cache_hit", 1)
            return self._user_scale[user_id]

        if is_enabled():
            sim_incr("sim.user_scale.cache_miss", 1)
        # sample data points for estimation
        if self.sample_k is None or self.sample_k >= self.num_items:
            item_indices = range(self.num_items)
        else:
            item_indices = self.rng.choice(self.num_items, size=self.sample_k, replace=False)

        try:
            item_indices_arr = np.asarray(list(item_indices), dtype=int)
            scores = self.rec_algo.predict(user_id, item_indices_arr)
        except Exception:
            # Fallback for rec models that only support scalar predict.
            scores = [self.rec_algo.predict(user_id, int(item_id)) for item_id in item_indices]

        mu = float(np.mean(scores))
        sigma = float(np.std(scores))

        if not np.isfinite(sigma) or sigma < self.min_sigma:
            sigma = self.min_sigma  # make sure sigma is non zero

        self._user_scale[user_id] = (mu, sigma)

        return mu, sigma

    def _user_params_normed(self, user_id: int) -> tuple[float, float, float]:
        """
        Converts raw thresholds to normed values using users mu (mean) and sigma (std).
        """
        mu, sigma = self._get_user_scale(user_id)
        b_u = self.global_bias # we have only global bias currently
        t1 = (self.rating_threshold - mu) / sigma + b_u
        t0 = (self.rating_threshold - self.delta_raw - mu) / sigma + b_u
        tau = self.tau_norm

        return t0, t1, tau

    def simulate_user_vote(self, user_id: int, item_id: int) -> int:
        # 1. get prediction
        score_raw = self.rec_algo.predict(user_id, item_id)

        # 2. add normalization
        mu, sigma = self._get_user_scale(user_id)
        s_norm = (score_raw - mu) / sigma

        # 3. get normed params
        t0_norm, t1_norm, tau_norm = self._user_params_normed(user_id)

        # 4. find probabilities
        p_like = self._sigmoid((s_norm - t1_norm) / tau_norm)      # chance for like
        p_dislike = self._sigmoid((t0_norm - s_norm) / tau_norm)   # chance for dislike
        p_neutral = 1.0 - p_like - p_dislike    # rest is neutral

        # 3. normalize
        p = np.clip([p_dislike, p_neutral, p_like], 0.0, 1.0)
        s = p.sum()

        if s == 0:
            p = np.array([0.4, 0.2, 0.4])  # fallback
        else:
            p = p / s

        # 4. sample
        return np.random.choice([-1, 0, 1], p=p)
