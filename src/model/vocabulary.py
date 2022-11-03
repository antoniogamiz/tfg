from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    words: List[str]
    _word_to_index: Dict[str, int]
    _index_to_word: Dict[int, str]

    def get_index_by_word(self, word: str) -> str | None:
        return self._word_to_index.get(word)

    @property
    def size(self):
        return len(self.words)

    @classmethod
    def from_words(cls, words: List[str]):
        word_to_index, index_to_word = dict(), dict()
        for index, word in enumerate(words):
            word_to_index[word] = index
            index_to_word[index] = word
        return cls(words=words, _word_to_index=word_to_index, _index_to_word=index_to_word)
