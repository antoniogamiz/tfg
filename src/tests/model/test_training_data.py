import unittest

from model.training_data import generate_training_data, TrainingData


class TrainingDataTestCase(unittest.TestCase):
    def test_given_corpus_when_generating_training_data_then_it_is_generated_correctly(self):
        corpus = ['words', 'some', 'words', 'a']

        actual_training_data = generate_training_data(window_size=3, corpus=corpus)

        expected_training_data = [
            TrainingData(target='words', context=['some', 'words', 'a']),
            TrainingData(target='some', context=['words', 'words', 'a']),
            TrainingData(target='words', context=['words', 'some', 'a']),
            TrainingData(target='a', context=['words', 'some', 'words']),
        ]
        self.assertEqual(expected_training_data, actual_training_data)
