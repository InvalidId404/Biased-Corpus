from konlpy.tag import Twitter
from random import choice
from random import random
from gensim.models.word2vec import Word2Vec


def read_data(fn):
    with open(fn, 'r', encoding='UTF-8') as f:
        data = [line.split('\t')[1:] for line in f.read().splitlines()]
        return data[1:]


kkma = Twitter()

train_data = read_data('D:\Programming\Dataset\\nsmc\\ratings_train.txt')
test_data = read_data('D:\Programming\Dataset\\nsmc\\ratings_test.txt')

positive_samples_train = []
negative_samples_train = []
for sample in train_data:
    if bool(int(sample[1])):
        positive_samples_train.append(kkma.morphs(sample[0]))
    else:
        negative_samples_train.append(kkma.morphs(sample[0]))


def extract(corpus_1, corpus_2, rate, sample):
    result = []
    for _ in range(sample):
        p = random()
        if p <= rate:
            result.append(choice(corpus_1))
        else:
            result.append(choice(corpus_2))
    return result


set_normal = extract(positive_samples_train, negative_samples_train, 0.5, 50000)
set_a = extract(positive_samples_train, negative_samples_train, 0.9, 50000)
set_a_2 = extract(positive_samples_train, negative_samples_train, 0.7, 50000)
set_b_2 = extract(positive_samples_train, negative_samples_train, 0.9, 50000)
set_b = extract(positive_samples_train, negative_samples_train, 0.1, 50000)
