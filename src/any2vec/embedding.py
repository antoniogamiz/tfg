from typing import Annotated

from numpy import ndarray

Embedding = Annotated[ndarray, "Wrapper of ndarray to ease understanding of Any2Vec model"]
