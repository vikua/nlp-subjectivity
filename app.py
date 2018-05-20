import os
import json
from flask import Flask, render_template, json, request

from nltk.tokenize import sent_tokenize
from models.estimators import  ULMFitEstimator


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # est = BaselineSubjectivityEstimator(os.path.join(dir_path, 'bin'))
    itos_path = os.path.join(dir_path, 'data', 'itos.pkl')
    model_path = os.path.join(dir_path, 'bin', 'models', 'golden_class_best.h5')
    est = ULMFitEstimator(model_path, itos_path)

    raw_sentences = sent_tokenize(text, language='russian')
    sentences = est.transform_sentences(raw_sentences)
    vectors = est.vectorize(sentences)
    predictions = est.predict(vectors)

    result = []
    for s, (o_score, s_score) in zip(raw_sentences, predictions):
        if o_score >= s_score:
            score = o_score
        else:
            score = -s_score
        result.append({
            'sentence': s,
            'score': float(score)
        })

    return json.dumps(result)


if __name__ == '__main__':
    app.run(debug=True)