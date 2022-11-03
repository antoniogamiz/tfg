from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    corpus: List[str]
    _word_to_index: Dict[str, int]
    _index_to_word: Dict[int, str]

    def get_index_by_word(self, word: str) -> str | None:
        return self._word_to_index.get(word)

    @property
    def size(self):
        return len(self.corpus)
