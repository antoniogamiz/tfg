import unittest
import os

from any2vec.anys.words import get_sentences_from_file, get_corpus_from_sentences, get_words_in_vocabulary, \
    generate_training_data
from any2vec.training_data import TrainingData


class WordsTestCase(unittest.TestCase):
    def test_get_sentences_from_file(self):
        test_file = 'test2.txt'
        test_file_contents = 'First sentence. Second sentence.'
        with open(test_file, 'w') as f:
            f.write(test_file_contents)

        actual_file_contents = get_sentences_from_file(test_file)

        os.remove(test_file)
        self.assertEqual(['First sentence', 'Second sentence'], actual_file_contents)

    def test_given_sentences_when_getting_the_corpus_then_is_correctly_generated(self):
        sentences = ["test sentence"]

        actual_corpus = get_corpus_from_sentences(sentences)

        expected_corpus = ["test", "sentence"]
        self.assertEqual(expected_corpus, actual_corpus)

    def test_given_corpus_when_extracting_the_words_then_they_are_extracted_correctly(self):
        corpus = ["some", "words", "repeated", "words"]

        actual_words = get_words_in_vocabulary(corpus)

        expected_words = ["repeated", "some", "words"]
        self.assertEqual(expected_words, actual_words)

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
