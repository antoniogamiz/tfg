from dataclasses import dataclass
from typing import TypeVar, Hashable

from numpy import ndarray

Data = TypeVar("Data", bound=Hashable)
OneHotEncoding = ndarray
Embedding = ndarray


@dataclass(frozen=True)
class LearningRate:
    value: float

    def __post_init__(self):
        if not 0 < self.value < 1:
            raise ValueError("Learning rate must be on the interval ]0,1[.")
