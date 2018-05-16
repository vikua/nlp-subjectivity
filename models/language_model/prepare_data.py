import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import tokenize_uk

from fastai.text import *
from sklearn.model_selection import train_test_split


class UKTokenizer(object):
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


def get_wiki_files(path):
    lm_files = []
    for d in os.listdir(path):
        wiki_files = os.listdir(os.path.join(path, d))
        for f in wiki_files:
            lm_files.append(os.path.join(path, d, f))

    print('Num wiki files:', len(lm_files))
    print('Top 10 files found: ', lm_files[0:10])

    return lm_files


def load_files_to_dataframe(files):
    texts = []
    for i in files:
        with open(i) as f:
            for line in f:
                texts.append(json.loads(line))
    return pd.DataFrame(texts)


def split_title_from_text(text):
    words = text.split("\n\n")
    if len(words) >= 2:
        return ''.join(words[1:])
    else:
        return ''.join(words)


def main(args):
    lm_files = get_wiki_files(args.wiki_files)
    df = load_files_to_dataframe(lm_files)

    print(df.head().to_string())
    print(df.shape)

    df['text'] = df['text'].apply(lambda x: split_title_from_text(x))
    df['len'] = df['text'].apply(lambda x: len(tokenize_uk.tokenize_words(x)))

    print('Overall number of tokens', df['len'].sum())
    print('Decreasing to ~100 million tokens')

    df = df[df['len'] > 600]

    print('New number of tokens', df['len'].sum())

    df['labels'] = 0
    df = df[['labels', 'text']]

    tokens = UKTokenizer().proc_all_mp(partition_by_cores(df['text'].values))
    labels = list(df['labels'].values.astype(np.int64))

    tokens_trn, tokens_val, labels_trn, labels_val = train_test_split(tokens, labels,
                                                                      test_size=0.1,
                                                                      random_state=1234,
                                                                      shuffle=True)

    # limiting vocabulary to ignore rare words
    freq = Counter(p for o in tokens_trn for p in o)

    itos = [o for o, c in freq.most_common(args.max_vocab) if c > args.min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

    trn_lm = np.array([[stoi[o] for o in p] for p in tokens_trn])
    val_lm = np.array([[stoi[o] for o in p] for p in tokens_val])

    np.save(os.path.join(args.output_dir, 'trn_ids.npy'), trn_lm)
    np.save(os.path.join(args.output_dir, 'val_ids.npy'), val_lm)

    with open(os.path.join(args.output_dir, 'itos.pkl'), 'wb') as f:
        pickle.dump(itos, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ukrainian LM data preparation')
    parser.add_argument('--wiki_files', dest='wiki_files', type=str,
                        help='Path to directory with wiki json files created '
                             'by https://github.com/attardi/wikiextractor.git')
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        help='Dir to save results to (numpy arrays)')
    parser.add_argument('--max_vocab', type=int, dest='max_vocab', default=60000,
                        help='Vocabulary limit to ignore rare words')
    parser.add_argument('--min_freq', type=int, dest='min_freq', default=2)
    args = parser.parse_args()
    main(args)
