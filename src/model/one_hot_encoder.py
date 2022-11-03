from typing import List

import numpy as np

from model.vocabulary import Vocabulary


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
