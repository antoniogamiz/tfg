from anys.tropes import get_unique_tropes, get_tropes_from_file
from vocabulary import Vocabulary

data = get_tropes_from_file('./datasets/sample.json')
tropes = get_unique_tropes(data)
vocabulary = Vocabulary.from_data_list(tropes)

print(vocabulary.data[0:5])
