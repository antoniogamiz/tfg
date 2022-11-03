import unittest
import os


from src.model.data import read_text_file, process_sentences, get_sentences_from_file


class DataTestCase(unittest.TestCase):
    def test_read_text_file(self):
        test_file = 'test.txt'
        test_file_contents = 'contents'
        with open(test_file, 'w') as f:
            f.write(test_file_contents)

        actual_file_contents = read_text_file(test_file)

        os.remove(test_file)
        self.assertEqual(test_file_contents, actual_file_contents)

    def test_get_sentences_from_file(self):
        test_file = 'test2.txt'
        test_file_contents = 'First sentence. Second sentence.'
        with open(test_file, 'w') as f:
            f.write(test_file_contents)

        actual_file_contents = get_sentences_from_file(test_file)

        os.remove(test_file)
        self.assertEqual(['First sentence', 'Second sentence'], actual_file_contents)

    def test_process_sentences(self):
        sentences = ["test sentence with a repeated word word"]

        word_to_index, index_to_word, corpus = process_sentences(sentences)

        expected_word_to_index = {'a': 0, 'repeated': 1, 'sentence': 2, 'test': 3, 'with': 4, 'word': 5}
        self.assertDictEqual(expected_word_to_index, word_to_index)
        expected_index_to_word = {0: 'a', 1: 'repeated', 2: 'sentence', 3: 'test', 4: 'with', 5: 'word'}
        self.assertDictEqual(expected_index_to_word, index_to_word)
        expected_corpus = ['test', 'sentence', 'with', 'a', 'repeated', 'word', 'word']
        self.assertEqual(expected_corpus, corpus)
