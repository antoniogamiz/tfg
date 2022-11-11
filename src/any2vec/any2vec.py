from dataclasses import dataclass, field
from typing import List, Generic, Callable, Iterator

import numpy as np
from numpy import ndarray

from any2vec.data import Data, OneHotEncoding, LearningRate
from any2vec.one_hot_encoder import encoding_training_data_item
from any2vec.training_data import TrainingData
from any2vec.vocabulary import Vocabulary


@dataclass
class Any2Vec(Generic[Data]):
    vocabulary: Vocabulary[Data]
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
            for data in self.training_data():
                data = encoding_training_data_item(self.vocabulary, data)
                y_predicted, hidden_layer, u = self.forward_propagation(data.target)
                error = self.error(y_predicted, data.context)
                self.backward_propagation(data.target, hidden_layer, error)
                loss += self.loss(u, data.context)
            self.historic_loss.append(loss)

    def forward_propagation(self, target_word: OneHotEncoding):
        hidden_layer = np.dot(self.weight_input_hidden.T, target_word)
        u = np.dot(self.weight_hidden_output.T, hidden_layer)
        y_predicted = softmax(u)
        return y_predicted, hidden_layer, u

    def backward_propagation(self, target_word: OneHotEncoding, hidden_layer: ndarray, error: ndarray):
        delta_weight_input_hidden = np.outer(target_word, np.dot(self.weight_hidden_output, error.T))
        self.weight_input_hidden -= self.learning_rate.value * delta_weight_input_hidden

        delta_weight_hidden_output = np.outer(hidden_layer, error)
        self.weight_hidden_output -= self.learning_rate.value * delta_weight_hidden_output

    @staticmethod
    def error(y_predicted: ndarray, y_expected: OneHotEncoding):
        number_of_expected_context_words = len(np.where(y_expected == 1)[0])
        return y_predicted * number_of_expected_context_words - y_expected

    @staticmethod
    def loss(u: ndarray, y_expected: OneHotEncoding):
        first_part = - u[y_expected == 1].sum()
        number_of_words_expected_words = len(np.where(y_expected == 1)[0])
        second_part = number_of_words_expected_words * np.log(np.sum(np.exp(u)))
        return first_part + second_part


def softmax(x: ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
