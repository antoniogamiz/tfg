from dataclasses import dataclass
from typing import Hashable, Protocol

from numpy import ndarray


class Data(Protocol, Hashable):
    """
        Base class to represent Data in the Any2Vec model. The
        only requirement for is to be Hashable (needed to create
        OneHotEncodings
    """
    pass


OneHotEncoding = ndarray
Embedding = ndarray


@dataclass(frozen=True)
class LearningRate:
    value: float

    def __post_init__(self):
        if not 0 < self.value < 1:
            raise ValueError("Learning rate must be on the interval ]0,1[.")
