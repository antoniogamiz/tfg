from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from any2vec.data import Data
from any2vec.one_hot_encoding import OneHotEncoding
from any2vec.training_data import TrainingData
from any2vec.vocabulary import Vocabulary


@dataclass
class EncodedTrainingData:
    target: OneHotEncoding
    context: OneHotEncoding


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


def encoding_training_data_item(vocabulary: Vocabulary, training_data: TrainingData) -> EncodedTrainingData:
    return EncodedTrainingData(
        target=one_hot_encode_data(vocabulary, training_data.target),
        context=one_hot_encode_data_list(vocabulary, training_data.context)
    )


def one_hot_encode_data(vocabulary: Vocabulary, data: Data) -> OneHotEncoding:
    index_of_word = vocabulary.get_index_by_data(data)
    return OneHotEncoding(indexes=[index_of_word], real_size=vocabulary.size)


def one_hot_encode_data_list(vocabulary: Vocabulary, data: Sequence[Data]) -> OneHotEncoding:
    indexes = [vocabulary.get_index_by_data(word) for word in data]
    return OneHotEncoding(indexes=indexes, real_size=vocabulary.size)
