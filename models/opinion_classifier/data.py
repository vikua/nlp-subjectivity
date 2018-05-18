import re
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import tokenize_uk
import pymorphy2
import gensim

from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle


class VocabularyProcessor(object):

    def __init__(self, max_doc_length, vocabulary):
        self.max_doc_length = max_doc_length
        self.vocabulary = vocabulary

    def transform(self, sentences):
        result = []
        for sentence in sentences:
            indexes = np.zeros((self.max_doc_length,), dtype=np.int32)
            for i, token in enumerate(sentence):
                if i >= self.max_doc_length:
                    break
                indexes[i] = self.vocabulary.get(token, 0)
            result.append(indexes)
        return np.array(result)


class Word2VecEmbeddingsLoader(object):

    def __init__(self, embeddings_path, embedding_dimension):
        self._embeddings_path = embeddings_path
        self._embedding_dimension = embedding_dimension

        self.model = gensim.models.KeyedVectors.load_word2vec_format(self._embeddings_path,
                                                                binary=False)

        self._vocab = dict()
        for i, w in enumerate(self.model.index2entity):
            self._vocab[w] = i + 1

        self._vocabulary_size = len(self.model.index2entity) + 1
        self._embedding_matrix = np.zeros((self._vocabulary_size, self._embedding_dimension))
        for word, idx in self._vocab.items():
            self._embedding_matrix[idx] = self.model[word]

    @property
    def embedding_dimension(self):
        return self._embedding_dimension

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def embedding_matrix(self):
        return self._embedding_matrix


def get_data(files, feature_col, target_col):
    data = pd.concat([pd.read_csv(x) for x in files])
    data = down_sample(data)

    X = data[feature_col].apply(tokenize_uk.tokenize_words)
    y = data[target_col]

    return X, y


def down_sample(data):
    subj = data[data['y'] == 1]
    obj = data[data['y'] == 0]

    obj_d = resample(obj, replace=False, n_samples=len(subj), random_state=1234)

    return shuffle(pd.concat([obj_d, subj]))


def get_train_test_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=1234)


def lemmatize(words, morph):
    result_words = []
    for w in words:
        m = morph.parse(w)[0]
        if m.normal_form:
            result_words.append(m.normal_form)
    return result_words


def delete_stop_words(words, stop_words):
    result = []
    for w in words:
        if w not in stop_words:
            result.append(w)
    return result

def tokenizer(x): 
    return x