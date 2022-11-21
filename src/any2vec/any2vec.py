from dataclasses import dataclass, field
from typing import List, Callable, Iterator

import numpy as np
from numpy import ndarray

from any2vec.one_hot_encoding import OneHotEncoding
from any2vec.learning_rate import LearningRate
from any2vec.one_hot_encoder import encoding_training_data_item
from any2vec.training_data import TrainingData
from any2vec.vocabulary import Vocabulary


@dataclass
class Any2Vec:
    vocabulary: Vocabulary
    training_data: Callable[[], Iterator[TrainingData]]
    weight_input_hidden: ndarray = field(init=False)
    weight_hidden_output: ndarray = field(init=False)
    learning_rate: LearningRate
    embedding_size: int
    historic_loss: List[float] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.weight_input_hidden = np.random.uniform(-1, 1, (self.vocabulary.size, self.embedding_size))
        self.weight_hidden_output = np.random.uniform(-1, 1, (self.embedding_size, self.vocabulary.size))

    def run(self, epochs: int):
        previously_computed_epochs = len(self.historic_loss)
        for i in range(previously_computed_epochs, epochs):
            print(f"Epoch {i + 1}/{epochs}.")
            loss = 0
            j = 0
            for data in self.training_data():
                j += 1
                print(f"Epoch: {i} - {j/30838.0}")
                encoded_data = encoding_training_data_item(self.vocabulary, data)
                y_predicted, hidden_layer, u = self.forward_propagation(encoded_data.target)
                error = self.error(y_predicted, encoded_data.context)
                self.backward_propagation(encoded_data.target, hidden_layer, error)
                loss += self.loss(u, encoded_data.context)
            self.historic_loss.append(loss)

    def forward_propagation(self, target_word: OneHotEncoding):
        hidden_layer = target_word.matrix_product(self.weight_input_hidden.T)
        u = np.dot(self.weight_hidden_output.T, hidden_layer)
        y_predicted = softmax(u)
        return y_predicted, hidden_layer, u

    def backward_propagation(self, target_word: OneHotEncoding, hidden_layer: ndarray, error: ndarray):
        delta_weight_input_hidden = np.dot(self.weight_hidden_output, error.T) * self.learning_rate.value
        for index in target_word.indexes:
            self.weight_input_hidden[index] -= delta_weight_input_hidden

        delta_weight_hidden_output = np.outer(hidden_layer, error) * self.learning_rate.value
        self.weight_hidden_output -= delta_weight_hidden_output

    @staticmethod
    def error(y_predicted: ndarray, y_expected: OneHotEncoding):
        number_of_expected_context_words = len(y_expected.indexes)
        return y_expected.subtract(y_predicted * number_of_expected_context_words)

    @staticmethod
    def loss(u: ndarray, y_expected: OneHotEncoding):
        first_part = - sum(u[index] for index in y_expected.indexes)
        number_of_words_expected_words = len(y_expected.indexes)
        second_part = number_of_words_expected_words * np.log(np.sum(np.exp(u)))
        return first_part + second_part


def softmax(x: ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
