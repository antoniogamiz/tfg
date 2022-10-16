from typing import List, Dict
from functools import reduce
import numpy as np


EXCLUDE_WORDS = ['\n', '-', '']


def process_sentences(sentences: List[str]) -> [Dict[str, int], Dict[int, str], List[str]]:
    all_words_by_sentences = map(lambda sentence: sentence.split(' '), sentences)
    unfiltered_corpus = reduce(lambda a, b: a+b, all_words_by_sentences)
    corpus = filter(lambda word: word not in EXCLUDE_WORDS, unfiltered_corpus)
    lower_case_corpus = list(map(lambda word: word.lower(), corpus))
    all_unique_words = list(set(lower_case_corpus))
    all_unique_words.sort()

    word_to_index, index_to_word = dict(), dict()
    for index, word in enumerate(all_unique_words):
        word_to_index[word] = index
        index_to_word[index] = word

    return word_to_index, index_to_word, lower_case_corpus


def get_sentences_from_file(path: str) -> List[str]:
    corpus = read_text_file(path)
    corpus = corpus.split('.')[0:-1]
    return [sentence.strip() for sentence in corpus]


def read_text_file(path: str) -> str:
    file_contents = None
    with open(path) as f:
        file_contents = f.read()

    if isinstance(file_contents, bytes):
        return file_contents.decode('utf-8')

    return file_contents
