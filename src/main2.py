import csv
import numpy as np
from scipy import spatial

from any2vec.anys.tropes import get_unique_tropes, get_tropes_from_file
from any2vec.vocabulary import Vocabulary

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


def cosine_matrix(embedding, embeddings):
    cosines = []
    for index, _embedding in embeddings.items():
            cosine_similarity = 1 - spatial.distance.cosine(embedding, _embedding)
            cosines.append([index, cosine_similarity])
    return sorted(cosines, key=lambda d: d[1], reverse=True)

def main():
    data = get_tropes_from_file('./datasets/sample.json')
    tropes = get_unique_tropes(data)
    vocabulary = Vocabulary.from_data_list(tropes)
    # embeddings = read_embeddings('.w100_0.001_100-cache/weight_input_hidden.csv')
    embeddings = read_embeddings('.w100_0.001_100-cache/weight_input_hidden.csv')
    a = list(np.array(embeddings[5]) - np.array(embeddings[6]))
    print(f"{vocabulary.data[5]} - {vocabulary.data[6]} = {vocabulary.data[cosine_matrix(a, embeddings)[3][0]]}")


    # cosines = {i: [] for i in range(n)}
    # for i in range(n):
    #     for j in range(n):
    #         cosine_similarity = 1 - spatial.distance.cosine(embeddings[i], embeddings[j])
    #         cosines[i].append([vocabulary.data[j], round(cosine_similarity, 2)])
    #
    # for i in range(n):
    #     cosines[i] = sorted(cosines[i], key=lambda d: d[1], reverse=True)
    #
    # def print_line(data):
    #     line = f"{data[0]}"
    #     for i in range(1, len(data)):
    #         line += f" & {data[i]}"
    #     line += "\\\\"
    #     print(line)
    #     print("\\hline")
    #
    # print("\\begin{center}")
    # print("\\begin{table}")
    # print("\\resizebox{\\textwidth}{!}{%")
    # print("\\begin{tabular}{|c|c|c|}")
    # print("\\hline")
    # print_line(["Trope", *[i for i in range(2)]])
    # for index in range(3):
    #     print_line([vocabulary.data[index], *[f"{t[0]}({t[1]})" for t in cosines[index][1:3]]])
    # print("\\end{tabular}}")
    # print("\\caption{\\label{w}Tropos similares usando embeddings de tamaño }")
    # print("\\end{table}")
    # print("\\end{center}")


if __name__ == "__main__":
    main()
