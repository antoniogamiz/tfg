from dataclasses import dataclass
from typing import List

from model.data import Data


@dataclass
class TrainingData:
    target: Data
    context: List[Data]
