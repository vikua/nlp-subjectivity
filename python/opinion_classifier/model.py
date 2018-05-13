import tensorflow as tf


WORD_FEATURE = 'words'

LEARNING_RATE = 'learning_rate'

VOCAB_SIZE = 'vocabulary_size'
SEQUENCE_LENGTH = 'sequence_length'
EMBEDDING_DIM = 'embedding_dimension'


def embedding_model(features, labels, mode, params):
    word_vectors = tf.contrib.layers.embed_sequence(features[WORD_FEATURE],
                                                    vocab_size=params[VOCAB_SIZE],
                                                    embed_dim=params[EMBEDDING_DIM])

    embedding_mean = tf.reduce_mean(word_vectors, axis=1)

    hidden = tf.layers.dense(embedding_mean, units=128, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    logits = tf.layers.dense(hidden, units=2, activation=None)
    predictions = tf.argmax(logits, axis=1, name='predictions')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'class': predictions,
                                                             'prob': tf.nn.softmax(logits)})

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params[LEARNING_RATE])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions),
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def rnn_model(features, labels, mode, params):
    word_vectors = tf.contrib.layers.embed_sequence(features[WORD_FEATURE],
                                                    vocab_size=params[VOCAB_SIZE],
                                                    embed_dim=params[EMBEDDING_DIM])

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(params[EMBEDDING_DIM])
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, units=2, activation=None)
    predictions = tf.argmax(logits, axis=1, name='predictions')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'class': predictions,
                                                             'prob': tf.nn.softmax(logits)})

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params[LEARNING_RATE])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions),
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
