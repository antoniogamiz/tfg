from os import path
from signal import signal, SIGINT
from dataclasses import dataclass
from typing import Generic

from numpy import savetxt, asarray, loadtxt

from any2vec.any2vec import Any2Vec
from any2vec.data import Data
from utils.filesystem import create_directory_if_not_exists, delete_file


@dataclass
class ModelRunner(Generic[Data]):
    name: str
    model: Any2Vec[Data]

    def start(self, epochs: int):
        if path.exists(self.cache_directory_path):
            self.restore_model_state()
        self.set_exit_signal()
        self.model.run(epochs)
        self.save_model_state()

    def set_exit_signal(self):
        def _signal(*_):
            self.save_model_state()
            exit(0)
        signal(SIGINT, _signal)

    def restore_model_state(self):
        self.model.weight_input_hidden = self.read_weight_input_hidden()
        self.model.weight_hidden_output = self.read_weight_hidden_output()
        self.model.historic_loss = self.read_historic_loss()

    def read_weight_input_hidden(self):
        return loadtxt(self.weight_input_hidden_path, delimiter=',')

    def read_weight_hidden_output(self):
        return loadtxt(self.weight_hidden_output_path, delimiter=',')

    def read_historic_loss(self):
        return list(loadtxt(self.historic_loss_path, delimiter=','))

    def save_model_state(self):
        self.setup_cache_directory()
        self.save_weight_input_hidden()
        self.save_weight_hidden_output()
        self.save_historic_loss()

    def setup_cache_directory(self):
        create_directory_if_not_exists(self.cache_directory_path)
        delete_file(self.weight_input_hidden_path)
        delete_file(self.weight_hidden_output_path)

    def save_weight_input_hidden(self):
        savetxt(self.weight_input_hidden_path, self.model.weight_input_hidden, delimiter=',') # noqa

    def save_weight_hidden_output(self):
        savetxt(self.weight_hidden_output_path, self.model.weight_hidden_output, delimiter=',') # noqa

    def save_historic_loss(self):
        historic_loss = asarray(self.model.historic_loss)
        savetxt(self.historic_loss_path, historic_loss, delimiter=',') # noqa

    @property
    def weight_input_hidden_path(self):
        return f'{self.cache_directory_path}/weight_input_hidden.csv'

    @property
    def weight_hidden_output_path(self):
        return f'{self.cache_directory_path}/weight_hidden_output.csv'

    @property
    def historic_loss_path(self):
        return f'{self.cache_directory_path}/historic_loss.csv'

    @property
    def cache_directory_path(self):
        return f".{self.name.lower()}-cache"
