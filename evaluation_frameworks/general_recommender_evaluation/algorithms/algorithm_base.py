from typing import Dict, List, Optional, Set, overload
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from evaluation_frameworks.general_recommender_evaluation.iterators.top_k_iterator import TopKIterator

class RecAlgoBase(ABC):
    @overload
    def fit(self, ratings_df: pd.DataFrame) -> "RecAlgoBase": ...

    @overload
    def fit(self, ratings: csr_matrix, user_id_map: Dict[int, int], item_id_map: Dict[int, int]) -> "RecAlgoBase": ...

    @overload
    def predict(self, user_id: int, item_id: int) -> float: ...

# TODO: move somewhere else??
class RecAlgoIterator(ABC):
    @abstractmethod
    def top_k_iterator(self, user_id: int, exclude: Optional[Set[int]] = None) -> TopKIterator:
        ...

class RecAlgoVectorizable(RecAlgoBase):
    @abstractmethod
    def get_user_vector(self, user_id: int) -> np.ndarray: ...

    @abstractmethod
    def get_item_scores(self, user_vector: np.ndarray) -> np.ndarray: ...

class RecAlgoMappable(ABC):

    @abstractmethod
    def item_id_to_index(self, item_id: int) -> int:
        """Map external item_id to internal matrix index."""
        ...

    @abstractmethod
    def index_to_item_id(self, index: int) -> int:
        """Map internal matrix index back to external item_id."""
        ...

class RecAlgoCached(ABC):
    @abstractmethod
    def precalculate_scores(self, user_ids: List[int]) -> None: ...

    @abstractmethod
    def get_cached_prediction(self, user_id: int, item_id: int) -> float: ...

    @abstractmethod
    def clear_cached_scores(self, user_id: Optional[int] = -1) -> None: ...


class RecAlgoFull(RecAlgoVectorizable, RecAlgoMappable, RecAlgoCached, RecAlgoIterator):
    """Convenience interface for fully-featured recommenders."""
    pass


