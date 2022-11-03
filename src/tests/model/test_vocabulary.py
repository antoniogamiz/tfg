import unittest


from src.model.vocabulary import Vocabulary


class VocabularyTestCase(unittest.TestCase):
    def test_given_a_vocabulary_when_retrieving_the_index_of_a_word_then_it_is_retrieved_correctly(self):
        vocabulary = Vocabulary(
            corpus=['some', 'words'],
            _word_to_index={'some': 1, 'word': 2},
            _index_to_word={1: 'some', 2: 'word'}
        )

        actual_index = vocabulary.get_index_by_word(word='some')

        expected_index = 1
        self.assertEqual(expected_index, actual_index)
