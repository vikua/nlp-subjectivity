import os

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from model import *
from data import *


tf.app.flags.DEFINE_string('train_dir', '/tmp/tensorflow/opinion/classfier', 'TODO')
tf.app.flags.DEFINE_string('data_dir', '../../data', 'TODO')

tf.app.flags.DEFINE_integer('epochs', 1, 'TODO')
tf.app.flags.DEFINE_integer('batch_size', 128, 'TODO')
tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'TODO')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'TODO')

tf.app.flags.DEFINE_string('embeddings_path', None, 'TODO')

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
    if not FLAGS.data_dir:
        raise ValueError('data_dir not provided')

    #if tf.gfile.Exists(FLAGS.train_dir):
    #    raise ValueError('This folder already exists.')
    #tf.gfile.MakeDirs(FLAGS.train_dir)

    files = ['classification_data_obj.csv',
             'classification_data_processed.csv',
             'classification_data_subj.csv']
    files = [os.path.join(FLAGS.data_dir, x) for x in files]

    X, y = get_data(files, feature_col='sentence_uk', target_col='y')
    X = clean_data(X)
    x_train_orig, x_test_orig, y_train, y_test = get_train_test_split(X.values, y.values)

    max_doc_length = max([len(x) for x in x_train_orig])

    w2v = Word2VecEmbeddingsLoader(FLAGS.embeddings_path, FLAGS.embedding_dim)
    vocabulary_processor = VocabularyProcessor(max_doc_length, w2v.vocabulary)

    x_train_seq_len = np.array([len(x) for x in x_train_orig], dtype=np.int32)
    x_test_seq_len = np.array([len(x) for x in x_test_orig], dtype=np.int32)
    x_train = vocabulary_processor.transform(x_train_orig)
    x_test = vocabulary_processor.transform(x_test_orig)

    params = {
        VOCAB_SIZE: w2v.vocabulary_size,
        EMBEDDING_DIM: FLAGS.embedding_dim,
        EMBEDDING_MATRIX: w2v.embedding_matrix,
        SEQUENCE_LENGTH: max_doc_length,
        LEARNING_RATE: FLAGS.learning_rate,
    }
    classifier = tf.estimator.Estimator(model_fn=rnn_model, params=params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORD_FEATURE: x_train,
           SEQUENCE_LENGTH_FEATURE: x_train_seq_len},
        y=y_train,
        num_epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        shuffle=True,
    )
    classifier.train(input_fn=train_input_fn)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORD_FEATURE: x_test,
           SEQUENCE_LENGTH_FEATURE: x_test_seq_len},
        y=y_test,
        num_epochs=1,
        batch_size=y_test.shape[0],
        shuffle=False,
    )
    results = classifier.evaluate(input_fn=test_input_fn)
    print('Evaluation results: {}'.format(results))

    predictions = classifier.predict(input_fn=test_input_fn)
    y_pred = [p['class'] for p in predictions]
    print('Test F1 score: ', f1_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # df = pd.DataFrame({'y_true': y_test.tolist(), 'y_pred': y_pred, 'x': x_test_orig.tolist()})
    # df[['y_pred', 'y_true', 'x']].to_csv(os.path.join(FLAGS.data_dir, 'errors.csv'), index=False, header=True)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()