from dataclasses import dataclass
from typing import Sequence

from any2vec.data import Data


@dataclass(frozen=True)
class TrainingData:
    target: Data
    context: Sequence[Data]
