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


set_normal = extract(positive_samples_train, negative_samples_train, 0.5, 50000)
set_a = extract(positive_samples_train, negative_samples_train, 0.9, 50000)
set_a_2 = extract(positive_samples_train, negative_samples_train, 0.7, 50000)
set_b = extract(positive_samples_train, negative_samples_train, 0.1, 50000)
set_b_2 = extract(positive_samples_train, negative_samples_train, 0.3, 50000)

model_normal = Word2Vec(sentences=set_normal, window=3)
model_a = Word2Vec(sentences=set_a, window=3)
model_a_2 = Word2Vec(sentences=set_a_2, window=3)
model_b = Word2Vec(sentences=set_b, window=3)
model_b_2 = Word2Vec(sentences=set_b_2, window=3)

words_to_compare = ['초딩', '반전', '개', '매력']

result = lambda model: [model.most_similar(word, topn=5) for word in words_to_compare]
string = '{0}의 결과\n' \
         '1. {1}와 가까운 단어: {2}, {3}, {4}, {5}, {6}\n' \
         '2. {7}와 가까운 단어: {8}, {9}, {10}, {11}, {12}\n' \
         '3. {13}와 가까운 단어: {14}, {15}, {16}, {17}, {18}\n' \
         '4. {19}와 가까운 단어: {20}, {21}, {22}, {23}, {24}\n'


def print_result(model, model_name):
    experiment = result(model)
    arg = [model_name]
    for i, word in enumerate(words_to_compare):
        arg.append(word)
        for r in experiment[i]:
            arg.append(r[0])
    print(string.format(*arg))


print_result(model_normal, '대조군')
print_result(model_a, '실험군 a')
print_result(model_a_2, '실험군 a_2')
print_result(model_b, '실험군 b')
print_result(model_b_2, '실험군 b_2')



