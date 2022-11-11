from dataclasses import dataclass
from typing import List

import numpy as np

from model.data import OneHotEncoding, Data
from model.training_data import TrainingData
from model.vocabulary import Vocabulary


@dataclass
class EncodedTrainingData:
    target: OneHotEncoding
    context: OneHotEncoding

    def __eq__(self, other):
        if not isinstance(other, EncodedTrainingData):
            return NotImplemented
        return (self.target == other.target).all() and (self.context == other.context).all()


def encode_training_data(vocabulary: Vocabulary, training_data: List[TrainingData]) -> List[EncodedTrainingData]:
    encoded_training_data = []
    for data in training_data:
        encoded_training_data.append(
            EncodedTrainingData(
                target=one_hot_encode_data(vocabulary, data.target),
                context=one_hot_encode_data_list(vocabulary, data.context)
            )
        )
    return encoded_training_data


def encoding_training_data_item(vocabulary: Vocabulary, training_data: TrainingData):
    return EncodedTrainingData(
        target=one_hot_encode_data(vocabulary, training_data.target),
        context=one_hot_encode_data_list(vocabulary, training_data.context)
    )


def one_hot_encode_data(vocabulary: Vocabulary, data: Data) -> OneHotEncoding:
    vector = np.zeros(vocabulary.size)
    index_of_word = vocabulary.get_index_by_data(data)
    vector[index_of_word] = 1
    return vector


def one_hot_encode_data_list(vocabulary: Vocabulary, data: List[Data]) -> OneHotEncoding:
    vector = np.zeros(vocabulary.size)
    for word in data:
        index_of_word = vocabulary.get_index_by_data(word)
        vector[index_of_word] = 1
    return vector
