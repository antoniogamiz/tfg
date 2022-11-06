import os
import pickle
from signal import signal, SIGINT

from model.any2vec import Any2Vec
from model.anys.words import get_sentences_from_file, get_corpus_from_sentences, get_words_in_vocabulary
from model.data import LearningRate
from model.training_data import generate_training_data
from model.vocabulary import Vocabulary

SAVE_MODEL_STATE_TO = 'model.pickle'


def any2vec_with_words(embedding_size: int, learning_rate: LearningRate) -> Any2Vec:
    print("Running Any2Vec with words...")

    sentences = get_sentences_from_file('jef_archer.txt')
    corpus = get_corpus_from_sentences(sentences)
    words = get_words_in_vocabulary(corpus)
    vocabulary = Vocabulary.from_data_list(words)

    any2vec = Any2Vec[str](
        vocabulary=vocabulary,
        training_data=generate_training_data(window_size=2, corpus=corpus),
        learning_rate=learning_rate,
        embedding_size=embedding_size
    )

    return any2vec


def main():
    previously_run_model_exists = os.path.exists(SAVE_MODEL_STATE_TO)
    if previously_run_model_exists:
        with open(SAVE_MODEL_STATE_TO, 'rb') as f:
            model = pickle.load(f)
    else:
        model = any2vec_with_words(embedding_size=2, learning_rate=LearningRate(value=0.001))

    def save_model_state(*_):
        with open('model.pickle', 'wb') as f:
            pickle.dump(model, f)
        exit(0)

    signal(SIGINT, save_model_state)

    model.run(epochs=30000)


if __name__ == "__main__":
    main()
