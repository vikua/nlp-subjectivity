import re
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import tensorflow as tf
import tokenize_uk
import pymorphy2
import gensim

from sklearn.model_selection import train_test_split


def get_data(files, feature_col, target_col):
    data = pd.concat([pd.read_csv(x) for x in files])

    X = data[feature_col].apply(tokenize_uk.tokenize_words)
    y = data[target_col]

    return X, y


def clean_data(text):
    reg = re.compile("""[\"#$%&*\-+/:;<=>@^`~…\\(\\)⟨⟩{}\[\|\]‒–—―«»“”‘’№]""")
    result = text.apply(lambda sent: [re.sub(reg, '', x) for x in sent])
    result = result.apply(lambda sent: [x for x in sent if x.strip()])

    result = result.apply(lambda sent: [x.lower() for x in sent])

    return result


def lemmatize_sentence_with_pos(words, morph):
    result = []
    for w in words:
        m = morph.parse(w)[0]
        result.append('{}_{}'.format(m.normal_form, m.tag.POS))
    return result


def lemmatize_chunk(chunk, func=lemmatize_sentence_with_pos):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    return chunk.apply(lambda x: func(x, morph))


def lemmatize_with_pos(text):
    cpu_count = mp.cpu_count()

    print('Starting lemmatization in {} processes'.format(cpu_count))

    pool = mp.Pool(cpu_count)
    function = partial(lemmatize_chunk, func=lemmatize_sentence_with_pos)
    results = pool.map(function, np.array_split(text, cpu_count))
    pool.close()
    pool.join()

    return pd.concat(results)


def get_train_test_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=1234)


class VocabularyProcessor(object):

    def __init__(self, max_doc_length, vocabulary):
        self.max_doc_length = max_doc_length
        self.vocabulary = vocabulary

    def transform(self, sentences):
        result = []
        for sentence in sentences:
            indexes = np.zeros((self.max_doc_length,), dtype=np.int32)
            for i, token in enumerate(sentence):
                indexes[i] = self.vocabulary.get(token, 0)
            result.append(indexes)
        return np.array(result)
