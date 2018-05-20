import re
import html
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import tokenize_uk
import pymorphy2
import gensim

from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle

from fastai.text import *


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


class UKTokenizer():
    def __init__(self):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

    def sub_br(self,x):
        return self.re_br.sub("\n", x)

    def tokenize(self, x):
        return tokenize_uk.tokenize_words(self.sub_br(x))

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP,TOK_SENT,TOK_MIX = ' t_up ',' t_st ',' t_mx '
        res = []
        prev='.'
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2))
    #                 else [TOK_SENT,s.lower()] if (s.istitle() and re_word.search(prev))
                    else [s.lower()])
    #         if re_nonsp.search(s): prev = s
        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(UKTokenizer.replace_rep, s)
        s = self.re_word_rep.sub(UKTokenizer.replace_wrep, s)
        s = UKTokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.tokenize(s)

    @staticmethod
    def proc_all(ss):
        tok = UKTokenizer()
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss):
        ncpus = num_cpus()//2
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(UKTokenizer.proc_all, ss), [])


def get_data(files, feature_col, target_col, downsample=False, escapechar=None, convert=False):
    data = pd.concat([pd.read_csv(x, escapechar=escapechar) for x in files])

    if convert:
        o = data[data[target_col] == '0']
        s = data[data[target_col] == '1']
        data = pd.concat([o, s])
        data[target_col] = data[target_col].astype(int)

    if downsample:
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


def clean_data(text):
    reg = re.compile("""[\"#$%&*\-+/:;<=>@^`~…\\(\\)⟨⟩{}\[\|\]‒–—―«»“”‘’№]""")
    result = text.apply(lambda sent: [re.sub(reg, '', x) for x in sent])
    result = result.apply(lambda sent: [x for x in sent if x.strip()])

    result = result.apply(lambda sent: [x.lower() for x in sent])

    return result


re1 = re.compile(r'  +')
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    tok = UKTokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels