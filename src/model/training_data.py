from dataclasses import dataclass
from typing import List


@dataclass
class TrainingData:
    target: str
    context: List[str]


def generate_training_data(window_size: int, corpus: List[str]) -> List[TrainingData]:
    training_data = []
    corpus_size = len(corpus)
    for target_word_index, target_word in enumerate(corpus):
        context_indexes = _get_context_indexes(corpus_size, window_size, target_word_index)
        context_words = [corpus[index] for index in context_indexes]
        training_data.append(TrainingData(target=target_word, context=context_words))
    return training_data


def _get_context_indexes(corpus_size: int, window_size: int, target_word_index: int) -> List[int]:
    start = 0
    end = corpus_size - 1
    if target_word_index == start:
        return _trim_indexes(list(range(start + 1, window_size + 1)), length=corpus_size)

    if target_word_index == end:
        return _trim_indexes(list(range(end - window_size, end)), length=corpus_size)

    before_indexes = list(range(target_word_index - window_size, target_word_index))
    after_indexes = list(range(target_word_index + 1, target_word_index + window_size + 1))
    indexes = before_indexes + after_indexes
    return _trim_indexes(indexes, length=corpus_size)


def _trim_indexes(indexes: List[int], length: int) -> List[int]:
    f = filter(lambda x: 0 <= x < length, indexes)
    return list(f)
