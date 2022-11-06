from model.any2vec import Any2Vec
from model.anys.words import get_sentences_from_file, get_corpus_from_sentences, get_words_in_vocabulary
from model.data import LearningRate
from model.training_data import generate_training_data
from model.vocabulary import Vocabulary


def word_example(embedding_size: int, epochs: int, learning_rate: LearningRate):
    print("Running Any2Vec with words...")

    sentences = get_sentences_from_file('./datasets/jef_archer.txt')
    corpus = get_corpus_from_sentences(sentences)
    words = get_words_in_vocabulary(sentences)
    vocabulary = Vocabulary.from_data_list(words)

    any2vec = Any2Vec[str](
        vocabulary=vocabulary,
        training_data=generate_training_data(window_size=2, corpus=corpus),
        learning_rate=learning_rate,
        embedding_size=embedding_size
    )

    any2vec.run(epochs=epochs)


def main():
    word_example(embedding_size=2, epochs=10, learning_rate=LearningRate(value=0.001))
