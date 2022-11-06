import unittest

from numpy import array

from model.one_hot_encoder import one_hot_encode_word, one_hot_encode_words, encode_training_data, EncodedTrainingData
from model.training_data import TrainingData
from model.vocabulary import Vocabulary


class OneHotEncodersTestCase(unittest.TestCase):
    VOCABULARY = Vocabulary(
        data=['some', 'words'],
        _data_to_index={'some': 0, 'word': 1},
        _index_to_data={0: 'some', 1: 'word'}
    )

    def test_given_word_when_generating_its_encoding_then_it_is_generated_correctly(self):
        word = 'some'

        actual_encoding = one_hot_encode_word(vocabulary=self.VOCABULARY, word=word)

        expected_encoding = array([1, 0])
        self.assertTrue((expected_encoding == actual_encoding).all())

    def test_given_words_when_generating_their_encoding_then_it_is_generated_correctly(self):
        words = ['some', 'words']

        actual_encoding = one_hot_encode_words(vocabulary=self.VOCABULARY, words=words)

        expected_encoding = array([1, 1])
        self.assertTrue((expected_encoding == actual_encoding).all())

    def test_given_traning_data_when_encoding_it_then_it_is_encoded_correctly(self):
        training_data = [TrainingData(target='some', context=['some', 'words'])]

        actual_encoded_training_data = encode_training_data(vocabulary=self.VOCABULARY, training_data=training_data)

        expected_training_data = [
            EncodedTrainingData(target=array([1, 0]), context=array([1, 1]))
        ]
        self.assertEqual(expected_training_data, actual_encoded_training_data)
