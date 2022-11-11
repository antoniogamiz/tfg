from dataclasses import dataclass
from typing import List

from any2vec.data import Data


@dataclass
class TrainingData:
    target: Data
    context: List[Data]
