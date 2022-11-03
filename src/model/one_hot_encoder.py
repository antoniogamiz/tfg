from dataclasses import dataclass
from typing import List

import numpy as np
from numpy import ndarray

from model.training_data import TrainingData
from model.vocabulary import Vocabulary


@dataclass
class EncodedTrainingData:
    target: ndarray
    context: ndarray

    def __eq__(self, other: "EncodedTrainingData"):
        return (self.target == other.target).all() and (self.context == other.context).all()


def encode_training_data(vocabulary: Vocabulary, training_data: List[TrainingData]) -> List[EncodedTrainingData]:
    encoded_training_data = []
    for data in training_data:
        encoded_training_data.append(
            EncodedTrainingData(
                target=one_hot_encode_word(vocabulary, data.target),
                context=one_hot_encode_words(vocabulary, data.context)
            )
        )
    return encoded_training_data


def one_hot_encode_word(vocabulary: Vocabulary, word: str):
    vector = np.zeros(vocabulary.size)
    index_of_word = vocabulary.get_index_by_word(word)
    vector[index_of_word] = 1
    return vector


def one_hot_encode_words(vocabulary: Vocabulary, words: List[str]):
    vector = np.zeros(vocabulary.size)
    for word in words:
        index_of_word = vocabulary.get_index_by_word(word)
        vector[index_of_word] = 1
    return vector
