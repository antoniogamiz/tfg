from model.data import process_sentences, get_sentences_from_file


def main():
    sentences = get_sentences_from_file('../datasets/jeff_archer.txt')
    word_to_index, index_to_word, corpus = process_sentences(sentences)
    print(corpus)
