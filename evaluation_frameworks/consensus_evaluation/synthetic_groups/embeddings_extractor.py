"""
`embeddings_extractor` — thin adapter from trained CF models to **user embedding dicts**.

Consensus evaluation builds synthetic groups in **external user id** space (e.g.
MovieLens ids). ``GroupGenerator`` needs a mapping ``{user_id: dense_vector}``.
Different trainers store vectors under **internal row indices**; LightFM exposes
``model.user_embeddings`` indexed by internal user index.

``EmbeddingExtractor`` centralises that mapping so ``eval_dataset_preparation`` and
``model_train_load`` do not duplicate LightFM-specific indexing. New model families
can be added behind the same ``model_type`` switch later.

**Why it exists:** one place to convert “model object + id map” → the dict shape
expected downstream (FAISS neighbour search, percentile thresholds, etc.).
"""

from typing import Dict

import numpy as np


class EmbeddingExtractor:
    """Pull user latent vectors from a supported model into ``{external_user_id: vector}``."""

    def __init__(self, model_type: str = "lightfm"):
        self.model_type = model_type

    def extract_user_embeddings(self, model, user_id_map: Dict[int, int] = None) -> Dict[int, np.ndarray]:
        """
        Return user embeddings keyed by **external** user id.

        ``user_id_map`` maps LightFM internal user index → external user id (required
        for ``model_type="lightfm"`` when ids are not identity).
        """
        if self.model_type == "lightfm":
            return self._extract_lightfm_user_embeddings(model, user_id_map)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported.")

    def _extract_lightfm_user_embeddings(self, model, user_id_map):
        n_users = model.user_embeddings.shape[0]
        user_embeddings = {}

        for user_idx in range(n_users):
            user_id = user_id_map[user_idx]
            user_embeddings[user_id] = model.user_embeddings[user_idx]

        return user_embeddings
