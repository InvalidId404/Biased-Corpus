from konlpy.tag import Kkma
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


set_normal = extract(positive_samples_train, negative_samples_train, 0.5, 10000)
set_a = extract(positive_samples_train, negative_samples_train, 0.7, 10000)
set_b = extract(positive_samples_train, negative_samples_train, 0.3, 10000)

model_normal = Word2Vec(sentences=set_normal, window=3)
model_a = Word2Vec(sentences=set_a, window=3)
model_b = Word2Vec(sentences=set_b, window=3)

words_to_compare = ['눈물', '곰', '개', '충격']

result = lambda model: [model.most_similar(word, topn=5) for word in words_to_compare]
string = '%d) %s의 결과' \
         '1. %s와 가까운 단어: %s, %s, %s, %s, %s' \
         '2. %s와 가까운 단어: %s, %s, %s, %s, %s' \
         '3. %s와 가까운 단어: %s, %s, %s, %s, %s' \
         '4. %s와 가까운 단어: %s, %s, %s, %s, %s'


def print_result(model, time, model_name):
    experiment = result(model)
    arg = [time, model_name]
    for i, word in enumerate(words_to_compare):
        arg.append(word)
        arg.append(*experiment[i])
    print(string %arg)


print_result(model_normal, 1, '대조군')
print_result(model_a, 2, '실험군 a')
print_result(model_b, 3, '실험군 b')



