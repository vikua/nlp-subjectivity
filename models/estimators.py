import dill as pickle
import os
import re

import pymorphy2
import stop_words
import tokenize_uk as tk

from models.opinion_classifier.data import lemmatize, delete_stop_words


class BaselineSubjectivityEstimator(object):
    reg = re.compile("""[\"#$%&*\-+/:;<=>@^`~…\\(\\)⟨⟩{}\[\|\]‒–—―«»“”‘’№]""")

    def __init__(self, models_path):
        with open(os.path.join(models_path, 'bow_vect.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(models_path, 'baseline_clf.pkl'), 'rb') as f:
            self.clf = pickle.load(f)

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
