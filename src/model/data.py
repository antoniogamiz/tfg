from typing import TypeVar, Hashable

from numpy import ndarray

Data = TypeVar("Data", bound=Hashable)
OneHotEncoding = ndarray
Embedding = ndarray
