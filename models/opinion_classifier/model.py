import tensorflow as tf


WORD_FEATURE = 'words'
SEQUENCE_LENGTH_FEATURE = 'seq_len'

LEARNING_RATE = 'learning_rate'
DROPOUT_PROB = 'dropout_keep_prob'

VOCAB_SIZE = 'vocabulary_size'
SEQUENCE_LENGTH = 'sequence_length'
EMBEDDING_DIM = 'embedding_dimension'
EMBEDDING_MATRIX = 'embedding_matrix'


def nn_model(features, labels, mode, params):
    word_vectors = tf.contrib.layers.embed_sequence(features[WORD_FEATURE],
                                                    vocab_size=params[VOCAB_SIZE],
                                                    embed_dim=params[EMBEDDING_DIM],
                                                    initializer=tf.constant_initializer(params[EMBEDDING_MATRIX]),
                                                    trainable=False)

    embedding_sum = tf.reduce_sum(word_vectors, axis=1)

    hidden = tf.layers.dense(embedding_sum, units=128, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    dropout = tf.layers.dropout(hidden, rate=0.5,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(dropout, units=2)
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
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    word_vectors = tf.contrib.layers.embed_sequence(features[WORD_FEATURE],
                                                    vocab_size=params[VOCAB_SIZE],
                                                    embed_dim=params[EMBEDDING_DIM],
                                                    initializer=tf.constant_initializer(params[EMBEDDING_MATRIX]),
                                                    trainable=False)

    cell = tf.nn.rnn_cell.LSTMCell(params[EMBEDDING_DIM], state_is_tuple=True)

    if is_training:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)

    outputs, last_state = tf.nn.dynamic_rnn(cell, word_vectors,
                                            sequence_length=features[SEQUENCE_LENGTH_FEATURE],
                                            dtype=tf.float32)

    states = tf.concat([last_state.c, last_state.h], axis=1 )

    hidden = tf.layers.dense(states, units=300, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    dropout = tf.layers.dropout(hidden, rate=0.5, training=is_training)

    logits = tf.layers.dense(dropout, units=2, activation=None)
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


def bi_lstm_model(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    word_vectors = tf.contrib.layers.embed_sequence(features[WORD_FEATURE],
                                                    vocab_size=params[VOCAB_SIZE],
                                                    embed_dim=params[EMBEDDING_DIM],
                                                    initializer=tf.constant_initializer(params[EMBEDDING_MATRIX]),
                                                    trainable=False)

    def create_lstm_cell():
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(params[EMBEDDING_DIM], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(params[EMBEDDING_DIM], state_is_tuple=True)

        if is_training and False:
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        return fw_cell, bw_cell

    fw_cell, bw_cell = create_lstm_cell()

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_vectors,
                                                 sequence_length=features[SEQUENCE_LENGTH_FEATURE],
                                                 dtype=tf.float32)
    output_fw, output_bw = outputs

    output_fw = tf.transpose(output_fw, [1, 0, 2])
    output_bw = tf.transpose(output_bw, [1, 0, 2])

    last_fw = output_fw[-1]
    last_bw = output_bw[-1]

    states = tf.concat([last_fw, last_bw], axis=1)

    logits = tf.layers.dense(states, units=2, activation=None)
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
