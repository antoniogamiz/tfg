from itertools import chain

from model.any2vec import Any2Vec
from model.anys.tropes import get_tropes_from_file, get_unique_tropes, get_training_tropes
from model.anys.words import get_sentences_from_file, get_corpus_from_sentences, get_words_in_vocabulary, \
    generate_training_data
from model.data import LearningRate
from model.model_runner import ModelRunner
from model.vocabulary import Vocabulary


def any2vec_with_words(embedding_size: int, learning_rate: LearningRate) -> [str, Any2Vec]:
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

    return "Words2", any2vec


def any2vec_with_tropes(embedding_size: int, learning_rate: LearningRate) -> [str, Any2Vec]:
    print("Running Any2Vec with tropes...")

    data = get_tropes_from_file('./datasets/dataset.json')
    tropes = get_unique_tropes(data)
    vocabulary = Vocabulary.from_data_list(tropes)

    print(f"Found {vocabulary.size} tropes in dataset.")
    iterations = len(list(chain.from_iterable(data.values())))
    print(f"Number of iterations per epoch: {iterations}")

    any2vec = Any2Vec[str](
        vocabulary=vocabulary,
        training_data=lambda: get_training_tropes(data),
        learning_rate=learning_rate,
        embedding_size=embedding_size
    )

    return "Tropes", any2vec


def main():
    name, model = any2vec_with_tropes(embedding_size=2, learning_rate=LearningRate(0.001))
    runner = ModelRunner(
        name=name,
        model=model
    )
    runner.start(epochs=1)

if __name__ == "__main__":
    main()
