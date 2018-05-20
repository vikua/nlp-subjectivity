import argparse
import pickle
import collections

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.text import *

from data import *


def main(args):
    df = pd.read_csv(args.input_file, skiprows=1, header=None, chunksize=args.chunksize)

    texts, labels = get_all(df, 1)

    text_train, text_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2,
                                                                        random_state=1234, stratify=labels)

    with open(args.itos_path, 'rb') as f:
        itos = pickle.load(f)
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

    train_clas = np.array([[stoi[o] for o in p] for p in text_train])
    val_clas = np.array([[stoi[o] for o in p] for p in text_test])
    trn_labels = np.squeeze(np.array(labels_train))
    val_labels = np.squeeze(np.array(labels_test))

    bptt, em_sz, nh, nl = 70, 400, 1150, 3
    vs = len(itos)

    # select our optimizer
    # also pick a batch size as big as you can that doesn't run out of memory
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    bs = 48

    min_lbl = trn_labels.min()
    trn_labels -= min_lbl
    val_labels -= min_lbl
    c = int(trn_labels.max()) + 1

    trn_ds = TextDataset(train_clas, trn_labels)
    val_ds = TextDataset(val_clas, val_labels)

    # sort the docs based on size.
    # validation will be explicitly short -> long
    # training, which sorts loosely
    trn_samp = SortishSampler(train_clas, key=lambda x: len(train_clas[x]), bs=bs // 2)
    val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))

    # then we create our dataloaders as before but with a [sampler] parameter
    trn_dl = DataLoader(trn_ds, bs // 2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(args.lm_model_path, trn_dl, val_dl)

    # setup our dropout rates
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5
    m = get_rnn_classifer(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                          layers=[em_sz * 3, 50, c], drops=[dps[4], 0.1],
                          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

    # define our RNN learner
    learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip = 25.
    learn.metrics = [accuracy]

    # set our learning rate
    # we will use discriminative learning rates for different layers
    lr = 3e-3
    lrm = 2.6
    lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])

    # Now we load our language model from before
    # but freeze everything except the last layer
    lrs = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2])
    wd = 1e-7
    wd = 0
    learn.load_encoder(args.lm_model_name)
    learn.freeze_to(-1)

    learn.lr_find(lrs / 1000)

    # train last layer
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))
    learn.save('{}_0'.format(args.model_name))

    # train two last layers
    learn.load('{}_0'.format(args.model_name))
    learn.freeze_to(-2)
    learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))
    learn.save('{}_1'.format(args.model_name))

    learn.load('{}_1'.format(args.model_name))
    learn.unfreeze()
    learn.fit(lrs, 1, wds=wd, cycle_len=args.epochs, use_clr=(32, 10),
              best_save_name='{}_best'.format(args.model_name),
              cycle_save_name='{}_cycle'.format(args.model_name))
    learn.save('{}_final'.format(args.model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ULM fit retrained classifier')
    parser.add_argument('--input_file', dest='input_file', help='Input CSV file')
    parser.add_argument('--itos_path', dest='itos_path', help='Index to word pickle file')
    parser.add_argument('--lm_model_path', dest='lm_model_path', help='Path to trained ULMfit model')
    parser.add_argument('--lm_model_name', dest='lm_model_name', help='Name of trained ULMfit model')
    parser.add_argument('--chunksize', dest='chunksize', default=2000, type=int, help='Pandas chunk size')
    parser.add_argument('--epochs', dest='epochs', default=5, type=int, help='Num epochs')
    parser.add_argument('--model_name', dest='model_name', default='clas', help='Model name')

    args = parser.parse_args()
    main(args)
