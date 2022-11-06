from __future__ import annotations

from typing import List, Dict, Generic
from dataclasses import dataclass

from model.data import Data


@dataclass(frozen=True)
class Vocabulary(Generic[Data]):
    data: List[Data]
    _data_to_index: Dict[Data, int]
    _index_to_data: Dict[int, Data]

    def get_index_by_data(self, data: Data) -> int | None:
        return self._data_to_index.get(data)

    @property
    def size(self) -> int:
        return len(self.data)

    @classmethod
    def from_data_list(cls, data_list: List[Data]) -> Vocabulary:
        data_to_index, index_to_data = dict(), dict()
        for index, word in enumerate(data_list):
            data_to_index[word] = index
            index_to_data[index] = word
        return cls(data=data_list, _data_to_index=data_to_index, _index_to_data=index_to_data)
