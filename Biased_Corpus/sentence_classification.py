from InvalidNN import invalidnn_new as inv
from InvalidNN.utill import test
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
from random import choice
from random import random
from os.path import exists
import numpy as np


twit = Okt()

embedding_size = 100
sequence_length = 100


# Read Data
def read_data(fn):
    with open(fn, 'r', encoding='UTF-8') as f:
        data = [line.split('\t')[1:] for line in f.read().splitlines()]
        return data[1:]


train_data = read_data('D:\Programming\Dataset\\nsmc\\ratings_train.txt')
test_data = read_data('D:\Programming\Dataset\\nsmc\\ratings_test.txt')

positive_samples_train = []
negative_samples_train = []
if exists('nsmc.model'):
    model = Word2Vec.load('nsmc.model')
else:
    corpus = []
    for sentence in train_data+test_data:
        corpus.append(twit.morphs(sentence[0]))
        print(corpus[-1])
    model = Word2Vec(corpus, min_count=1, workers=4)
    model.save('nsmc.model')

for sample in train_data:
    m = [[model[w] for w in twit.morphs(sample[0])], [0.01, 0.99] if bool(int(sample[1])) else [0.99, 0.01]]
    if len(m[0]) < sequence_length:
        m[0] += [np.zeros([embedding_size]) for _ in range(sequence_length-len(m[0]))]
    m[0] = np.array(m[0]).reshape(sequence_length, embedding_size, 1)
    if bool(int(sample[1])):
        positive_samples_train.append(m)
    else:
        negative_samples_train.append(m)


def extract(corpus_1, corpus_2, rate, sample):
    result = []
    for _ in range(sample):
        p = random()
        if p <= rate:
            result.append(choice(corpus_1))
        else:
            result.append(choice(corpus_2))
    return result


train_set = extract(positive_samples_train, negative_samples_train, 0.5, 100000)
set_normal = extract(positive_samples_train, negative_samples_train, 0.5, 10000)
set_a = extract(positive_samples_train, negative_samples_train, 0.9, 10000)
set_b = extract(positive_samples_train, negative_samples_train, 0.1, 10000)

filters = [
    ('Bigram_1', 2),
    ('Bigram_2', 2),
    ('Trigram', 3)
]

layers = [
    [[inv.Conv2D(f[0], 'relu', 128, [f[1], embedding_size], [1, 1], 'valid'),
        inv.Pooling(f[0], 'max', [1, 1], [sequence_length-f[1], 1], 'valid')] for f in filters]
] + [inv.Dense('Dense', 'softmax', 2)]

classifier = inv.NeuralNetwork(layers, [None, sequence_length, embedding_size, 1])

result = []
for e, experiment in enumerate((set_normal, set_a, set_b)):
    result.append([e, test.test_model(classifier, experiment)])
print(result)

"""
classifier.train(
    training_dataset=train_set,
    batch_size=100,
    loss_fn='least-square',
    optimizer='gradient-descent',
    learning_rate=0.05,
    epoch=10000,
    drop_p=1.,
    model_path='./model',
    summary_path='./summary'
)

print('JOBS FINISHED')
"""