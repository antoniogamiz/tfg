import json

from any2vec.anys.tropes import get_tropes_from_file, get_unique_tropes


def read_original_dataset():
    data = get_tropes_from_file('./datasets/dataset.json')
    # AmericanFilms is not a trope, the scraper is wrong
    tropes_dict = {}
    for film, tropes in data.items():
        tropes_dict[film] = list(filter(lambda t: t != "AmericanFilms", tropes))
    return tropes_dict


def tropes_with_less_than_occurrence(tropes_frequency, occurrence):
    return set([trope for trope, frequency in tropes_frequency.items() if frequency < occurrence])


def print_dataset_statistics(dataset):
    unique_tropes = get_unique_tropes(dataset)
    tropes_frequency = get_trope_frequency_dict(dataset)

    print(f"Number of films: {len(dataset.keys())}")
    print(f"Number of unique tropes: {len(unique_tropes)}")
    most_popular_trope = max(tropes_frequency, key=tropes_frequency.get)
    least_popular_trope = min(tropes_frequency, key=tropes_frequency.get)
    print(f"Most popular trope: {most_popular_trope}: {tropes_frequency[most_popular_trope]}")
    print(f"Least popular trope: {least_popular_trope}: {tropes_frequency[least_popular_trope]}")

    print(f"Tropes with one occurrence: {len(tropes_with_less_than_occurrence(tropes_frequency, 1))}")

    print(f"Tropes with 40 occurrence: {len(tropes_with_less_than_occurrence(tropes_frequency, 40))}")


def get_trope_frequency_dict(dataset):
    unique_tropes = get_unique_tropes(dataset)
    tropes_frequency = {trope: 0 for trope in unique_tropes}
    for film, tropes in dataset.items():
        for trope in tropes:
            tropes_frequency[trope] += 1
    return tropes_frequency


def filter_dataset(dataset, minimum_number_of_tropes, minimum_occurrence):
    tropes_frequency = get_trope_frequency_dict(dataset)
    discarded_tropes = tropes_with_less_than_occurrence(tropes_frequency, minimum_occurrence)
    filtered_dataset = {}
    for film, tropes in dataset.items():
        filtered_tropes = list(filter(lambda x: x not in discarded_tropes, tropes))
        if len(filtered_tropes) > minimum_number_of_tropes:
            filtered_dataset[film] = filtered_tropes
    return filtered_dataset


tropes_dictionary = read_original_dataset()
print_dataset_statistics(tropes_dictionary)
print("=====================================")
new_dataset = filter_dataset(tropes_dictionary, 40, 500)
print_dataset_statistics(new_dataset)

with open("./datasets/sample.json", "w") as outfile:
    json.dump(new_dataset, outfile)
