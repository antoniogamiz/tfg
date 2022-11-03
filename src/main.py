import numpy as np

from model.any2vec import Any2Vec
from model.data import process_sentences, get_sentences_from_file
from model.training_data import generate_training_data
from model.vocabulary import Vocabulary


def main():
    embedding_size = 2
    sentences = get_sentences_from_file('./datasets/jef_archer.txt')
    word_to_index, index_to_word, corpus = process_sentences(sentences)

    vocabulary = Vocabulary(words=list(set(corpus)), _word_to_index=word_to_index, _index_to_word=index_to_word)
    any2vec = Any2Vec(
        vocabulary=vocabulary,
        training_data=generate_training_data(window_size=2, corpus=corpus),
        weight_input_hidden=np.random.uniform(-1, 1, (vocabulary.size, embedding_size)),
        weight_hidden_output=np.random.uniform(-1, 1, (embedding_size, vocabulary.size)),
        learning_rate=0.001
    )

    any2vec.run(epochs=100)


if __name__ == "__main__":
    main()
