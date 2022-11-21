import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from any2vec.anys.tropes import get_tropes_from_file, get_unique_tropes
from any2vec.vocabulary import Vocabulary


def word_similarity_scatter_plot(vocabulary, embeddings):
    labels = []
    tokens = []

    for key, value in vocabulary._index_to_data.items():
        tokens.append(np.array(embeddings[key]))
        labels.append(value)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(np.array(tokens))

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # plt.figure(figsize=(5, 5))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


n = 173


def read_embeddings(path):
    embeddings = {i: [] for i in range(n)}
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            for value in row:
                embeddings[i].append(float(value))
            i += 1
    return embeddings


def main():
    data = get_tropes_from_file('./datasets/sample.json')
    tropes = get_unique_tropes(data)
    vocabulary = Vocabulary.from_data_list(tropes)
    embeddings = read_embeddings('.w100_0.001_100-cache/weight_input_hidden.csv')
    # embeddings = read_embeddings('.w50_0.001_100-cache/weight_input_hidden.csv')
    word_similarity_scatter_plot(vocabulary, embeddings)
    plt.legend()

    # plt.set(xlim=(0.45, 0.55), ylim=(0.4, 0.6), autoscale_on=False,
    #     title='Zoom window')
    plt.show()

if __name__ == "__main__":
    main()
