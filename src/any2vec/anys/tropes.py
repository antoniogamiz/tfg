import json
from itertools import chain
from typing import Dict, List, Iterator

from any2vec.training_data import TrainingData
from utils.filesystem import read_text_file


def get_tropes_from_file(path: str) -> Dict:
    return json.loads(read_text_file(path))


def get_unique_tropes(data: Dict) -> List[str]:
    return list(set(chain.from_iterable(data.values())))


def get_training_tropes(data: Dict) -> Iterator[TrainingData]:
    for values in data.values():
        for index, value in enumerate(values):
            yield TrainingData(
                target=value,
                context=values[0:index] + values[index + 1:]
            )
