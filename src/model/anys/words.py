from typing import List, Dict
from functools import reduce

from utils.filesystem import read_text_file

EXCLUDE_WORDS = ['\n', '-', '']


def get_corpus_from_sentences(sentences: List[str]) -> [Dict[str, int], Dict[int, str], List[str]]:
    all_words_by_sentences = map(lambda sentence: sentence.split(' '), sentences)
    unfiltered_corpus = reduce(lambda a, b: a + b, all_words_by_sentences)
    corpus = filter(lambda _word: _word not in EXCLUDE_WORDS, unfiltered_corpus)
    return list(map(lambda _word: _word.lower(), corpus))


def get_words_in_vocabulary(corpus: List[str]):
    all_unique_words = list(set(corpus))
    all_unique_words.sort()
    return all_unique_words


def get_sentences_from_file(path: str) -> List[str]:
    corpus = read_text_file(path)
    corpus = corpus.split('.')[0:-1]
    return [sentence.strip() for sentence in corpus]
