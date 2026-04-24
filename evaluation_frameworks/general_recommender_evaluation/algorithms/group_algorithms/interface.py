from typing import Dict, List, Optional, Set, Tuple, Union, overload
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from evaluation_frameworks.general_recommender_evaluation.iterators.top_k_iterator import TopKIterator

class RecAlgoGroupIterator(ABC):
    @abstractmethod
    def top_k_iterator(self, user_ids: List[int], exclude: Optional[Set[int]] = None) -> TopKIterator:
        ...

class RecAlgoGroupAggregated(RecAlgoGroupIterator):
    @abstractmethod
    def top_k_iterator(self, user_ids: List[int], method: str = 'mean', exclude: Optional[Set[int]] = None) -> TopKIterator:
        ...


#class RecAlgoUpdatable(ABC):
#    """
#    Interface for recommenders that support dynamic updates to internal state,
#    either for single users or groups of users.
#    """
#
#    @abstractmethod
#    def update_group_with_votes(self, user_ids: List[int], votes: Dict[int, float]) -> None:
#        ...
#
#    @abstractmethod
#    def reset_group_state(self, user_ids: List[int]) -> None:
#        ...

class RecAlgoUpdatable(ABC):
    """
    Interface for recommenders that support dynamic updates to internal state,
    either for single users or groups of users.
    """

    @abstractmethod
    def update_one(self, user_id: int, item_id: int, value: float) -> None:
        ...

    @abstractmethod
    def update_many(self, user_item_interactions: Dict[int, Dict[int, float]]) -> None:
        ...

    @abstractmethod
    def reset_state(self) -> None:
        ...