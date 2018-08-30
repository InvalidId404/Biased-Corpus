from InvalidNN import invalidnn_new as inv


embedding_size = 0
vocab_size = 0
sequence_length = 0

filters = [
    ('Bigram_1', 2),
    ('Bigram_2', 2),
    ('Trigram', 3)
]
network = [
    [[inv.Conv2D(name=filter[0], activate_fn='relu', filter_shape=[filter[1], sequence_length], [1, 1], 'same'), inv.Pooling,
         inv.Pooling(filter[0], 'max', [3, 3], [filter[0], sequence_length], 'valid')] for filter in filters],
    inv.Dense('Dense_1', )
]
