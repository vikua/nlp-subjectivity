import dill
import os
import re
import pickle
import collections

import pymorphy2
import stop_words
import tokenize_uk as tk

from fastai.text import *
from fastai.model import predict
from fastai.core import T, V, to_np
from fastai.torch_imports import load_model
from torch.nn import Softmax

from models.opinion_classifier.data import lemmatize, delete_stop_words, UKTokenizer


class BaselineSubjectivityEstimator(object):
    reg = re.compile("""[\"#$%&*\-+/:;<=>@^`~…\\(\\)⟨⟩{}\[\|\]‒–—―«»“”‘’№]""")

    def __init__(self, models_path):
        with open(os.path.join(models_path, 'bow_vect.pkl'), 'rb') as f:
            self.vectorizer = dill.load(f)
        with open(os.path.join(models_path, 'baseline_clf.pkl'), 'rb') as f:
            self.clf = dill.load(f)

        self.morph = pymorphy2.MorphAnalyzer(lang='uk')
        self.stop_words = stop_words.get_stop_words(language='uk')

    def transform_sentences(self, sentences):
        result = []
        for sentence in sentences:
            words = tk.tokenize_words(sentence)
            words = [re.sub(self.reg, '', x) for x in words]
            words = [x for x in words if x.strip()]
            words = [x.lower() for x in words]
            words = lemmatize(words, morph=self.morph)
            words = delete_stop_words(words, stop_words=self.stop_words)
            result.append(words)

        return result

    def vectorize(self, sentences):
        return self.vectorizer.transform(sentences)

    def predict(self, x):
        return self.clf.predict_proba(x)


class ULMFitEstimator(object):

    def __init__(self, models_path, itos_path):
        with open(itos_path, 'rb') as f:
            self.itos = pickle.load(f)
        self.stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(self.itos)})

        bptt, em_sz, nh, nl = 70, 400, 1150, 3
        vs = len(self.itos)
        c = 2

        dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5
        self.m = get_rnn_classifer(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                   layers=[em_sz * 3, 50, c], drops=[dps[4], 0.1],
                                   dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
        load_model(self.m, models_path)

    def transform_sentences(self, sentences):
        return [UKTokenizer().proc_text(x) for x in sentences]

    def vectorize(self, sentences):
        return np.array([[self.stoi[o] for o in p] for p in sentences])

    def predict(self, x):
        fake_labels = np.array([0] * len(x))

        ds = TextDataset(x, fake_labels)
        dl = DataLoader(ds, 1000, transpose=True, num_workers=1, pad_idx=1)

        preds = predict(self.m, dl)
        sm = Softmax()
        return to_np(sm(V(T(preds))))
