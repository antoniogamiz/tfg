from dataclasses import dataclass
from typing import Any, List

from numpy.core.records import ndarray
import numpy as np

from model.one_hot_encoder import encode_training_data
from model.training_data import TrainingData
from model.vocabulary import Vocabulary


@dataclass
class Any2Vec:
    vocabulary: Vocabulary
    training_data: List[TrainingData]
    weight_input_hidden: Any
    weight_hidden_output: Any
    learning_rate: float

    def run(self, epochs: int):
        encoded_training_data = encode_training_data(self.vocabulary, self.training_data)

        for i in range(epochs):
            print(f"Epoch {i}. Remaining {epochs - i - 1}")
            for data in encoded_training_data:
                y_predicted, hidden_layer, u = self.forward_propagation(data.target)
                error = self.error(y_predicted, data.context)
                self.backward_propagation(data.target, hidden_layer, error)

    def forward_propagation(self, target_word: ndarray):
        hidden_layer = np.dot(self.weight_input_hidden.T, target_word)
        u = np.dot(self.weight_hidden_output.T, hidden_layer)
        y_predicted = softmax(u)
        return y_predicted, hidden_layer, u

    def backward_propagation(self, target_word: ndarray, hidden_layer: ndarray, error: ndarray):
        delta_weight_input_hidden = np.outer(target_word, np.dot(self.weight_hidden_output, error.T))
        self.weight_input_hidden -= self.learning_rate * delta_weight_input_hidden

        delta_weight_hidden_output = np.outer(hidden_layer, error)
        self.weight_hidden_output -= self.learning_rate * delta_weight_hidden_output

    @staticmethod
    def error(y_predicted: ndarray, y_expected: ndarray):
        number_of_expected_context_words = len(np.where(y_expected == 1)[0])
        return y_predicted * number_of_expected_context_words - y_expected

    @staticmethod
    def loss(u: ndarray, y_expected: ndarray):
        first_part = - u[y_expected == 1].sum()
        number_of_words_expected_words = len(np.where(y_expected == 1)[0])
        second_part = number_of_words_expected_words * np.log(np.sum(np.exp(u)))
        return first_part + second_part


def softmax(x: ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()