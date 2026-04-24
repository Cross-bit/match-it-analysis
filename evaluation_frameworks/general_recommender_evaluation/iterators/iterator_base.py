from abc import ABC, abstractmethod
from typing import Iterator


class BaseRecommendationIterator(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> int:
        raise NotImplementedError