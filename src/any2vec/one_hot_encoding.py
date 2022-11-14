from dataclasses import dataclass
from typing import Sequence, List

from numpy import ndarray
import numpy as np


@dataclass(frozen=True)
class OneHotEncoding:
    """
        Represents a one-hot encoded array. Usually, a hot-encoded array
        looks something like this: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], that is,
        all zeros except for one place where a one is stored. For efficiency reasons,
        we use one-hot vectors that can have more than one "1". That is, [0, 1, 1, 0, 1].

        In addition, in any2vec the vectors are as large as the vocabulary size. That means that
        if you have 37k different terms in your vocabulary (like in the tropes vocabulary) you
        will have really big vectors that are mostly zeros. For that reason, we only store the indexes
        of the vector where a 1 is stored.

        So, if you have the following vector: [0, 1, 0, 0, 1, 0], then it will stored internally as
        OneHotEncoding(indexes=[1, 4]).
    """
    indexes: List[int]
    real_size: int

    def matrix_product(self, other: ndarray) -> ndarray:
        if not isinstance(other, ndarray):
            raise ValueError("Use ndarray if you want to compute the product of a OneHotEncoding vector")
        if other.shape[1] != self.real_size:
            raise ValueError("Invalid dimensions to compute the product")
        result = np.zeros(other.shape[0])
        for i in range(other.shape[0]):
            result[i] = sum([other[i][index] for index in self.indexes])
        return result

    def subtract(self, other: ndarray) -> ndarray:
        for index in self.indexes:
            other[index] -= 1
        return other
