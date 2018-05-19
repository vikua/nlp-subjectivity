import argparse
import pickle
import os

import numpy as np

from fastai.text import *


def main(args):
    trn_lm = np.load(os.path.join(args.input_dir, 'trn_ids.npy'))
    val_lm = np.load(os.path.join(args.input_dir, 'val_ids.npy'))

    with open(os.path.join(args.input_dir, 'itos.pkl'), 'rb') as f:
        itos = pickle.load(f)

    vs = len(itos)

    em_sz, nh, nl = 400, 1150, 3

    wd = 1e-7
    bptt = 70
    bs = 52
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
    md = LanguageModelData(args.model_dir, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7

    learner = md.get_model(opt_fn, em_sz, nh, nl,
                           dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

    learner.metrics = [accuracy]
    learner.clip = 0.2
    learner.unfreeze()

    lr = 1e-3
    lrs = lr

    learner.fit(lrs, 1, wds=wd, use_clr=(32, 2, 0.95, 0.85), cycle_len=args.epochs,
                cycle_save_name='lm_ukrainian_cycle_v3', best_save_name='lm_ukrainian_best_v3')

    learner.save('lm_ukrainian_v3')
    learner.save_encoder('lm_ukrainian_encoder_v3')

    # learner.load('lm_ukrainian')
    # print(learner.lr_find(start_lr=lrs / 10, end_lr=lrs * 10, linear=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language model for Ukrainian')
    parser.add_argument('--input_dir', dest='input_dir', help='Dir with numpy arrays')
    parser.add_argument('--model_dir', dest='model_dir', help='Dir to save model to')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='Num epochs')

    args = parser.parse_args()
    main(args)