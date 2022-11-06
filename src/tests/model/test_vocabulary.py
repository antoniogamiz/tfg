import unittest


from model.vocabulary import Vocabulary


class VocabularyTestCase(unittest.TestCase):
    def test_given_a_vocabulary_when_retrieving_the_index_of_a_word_then_it_is_retrieved_correctly(self):
        vocabulary = Vocabulary(
            data=['some', 'words'],
            _data_to_index={'some': 1, 'word': 2},
            _index_to_data={1: 'some', 2: 'word'}
        )

        actual_index = vocabulary.get_index_by_data(data='some')

        expected_index = 1
        self.assertEqual(expected_index, actual_index)

    def test_given_a_vocabulary_when_getting_size_then_is_computed_correctly(self):
        vocabulary = Vocabulary(
            data=['some', 'words'],
            _data_to_index={'some': 1, 'word': 2},
            _index_to_data={1: 'some', 2: 'word'}
        )

        actual_size = vocabulary.size

        expected_size = 2
        self.assertEqual(expected_size, actual_size)

    def test_given_words_when_generating_the_vocabulary_from_them_then_is_generated(self):
        words = ['some', 'words']

        actual_vocabulary = Vocabulary.from_data_list(words)

        expected_vocabulary = Vocabulary(
            data=['some', 'words'],
            _data_to_index={'some': 0, 'words': 1},
            _index_to_data={0: 'some', 1: 'words'}
        )
        self.assertEqual(expected_vocabulary, actual_vocabulary)
