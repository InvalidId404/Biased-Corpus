# Skip-Gram
import tensorflow as tf
from konlpy.tag import Kkma


def one_hot(value, length):
    return [0.99 if i == value else 0.01 for i in range(length)]


def word2vec(corpus, embedding_size, window):
    vocabulary = []
    training_dataset = []
    for sentence in corpus:
        kkma = Kkma()
        sentence = kkma.morphs(sentence)
        vocabulary += sentence
        training_dataset.append(sentence)
    vocabulary = {i: word for i, word in enumerate(list(set(vocabulary)))}
    vocabulary_byvalue = {word: key for key, word in vocabulary.items()}
    vocab_size = len(vocabulary)

    # 문장에서 배치 뽑아내서 학습
    def make_batch(sentence, window):
        result = []
        for w, word in enumerate(sentence):
            batch = []
            for step in range(1, window+1):
                if (w-step) >= 0:
                    batch.append([one_hot(vocabulary_byvalue[word], vocab_size), one_hot(vocabulary_byvalue[sentence[w-step]], vocab_size)])
                if (w+step) <= len(sentence)-1:
                    batch.append([one_hot(vocabulary_byvalue[word], vocab_size), one_hot(vocabulary_byvalue[sentence[w+step]], vocab_size)])
            result.append(batch)
        return result
    training_dataset = sum([make_batch(sen, window) for sen in training_dataset], [])

    input_vec = tf.placeholder(tf.float32, [None, vocab_size])
    output_vec = tf.placeholder(tf.float32, [None, vocab_size])

    embeddings = tf.Variable(
        tf.random_normal([vocab_size, embedding_size])
    )
    output_weights = tf.Variable(
        tf.random_normal([embedding_size, vocab_size]) / tf.sqrt(float(embedding_size))
    )
    output_bias = tf.Variable(tf.random_normal([vocab_size]))
    loss = tf.reduce_mean(
        tf.nn.nce_loss(output_weights, output_bias, output_vec, input_vec, 5, vocab_size, num_sampled=2 )
    )
    train_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for batch in training_dataset:
            x_batch = [b[0] for b in batch]
            y_batch = [b[1] for b in batch]
            sess.run(train_step, feed_dict={input_vec: x_batch, output_vec: y_batch})

            lookup_table = sess.run(embeddings)

    return lookup_table


def main():
    test_corpus = ['그럴려고 그랬어 돌아가려고',
                   '너의 차가움엔 그래 다 이유 있었던거야',
                   '나를 만지는 너의 손길 없어진 이제야',
                   '깨닫게 되었어 네맘 떠나간것을',
                   '설마~하는 그런 미련때문에',
                   '그래도 나는 나를 위로해',
                   '이제 이러는 내가 더 가여워',
                   '이제라도 널 지울꺼야 기억의 모두를']
    word2vec(test_corpus, 100, 2)


if __name__ == '__main__':
    main()
