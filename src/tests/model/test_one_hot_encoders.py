import unittest

from numpy import array

from src.model.one_hot_encoder import one_hot_encode_word, one_hot_encode_words
from src.model.vocabulary import Vocabulary


class OneHotEncodersTestCase(unittest.TestCase):
    VOCABULARY = Vocabulary(
        corpus=['some', 'words'],
        _word_to_index={'some': 0, 'word': 1},
        _index_to_word={1: 'some', 0: 'word'}
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
