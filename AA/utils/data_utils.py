import os
import random
import numpy as np

_vocabs = {}

# Assuming negative label is 0 and positive is 1
def get_raw_data(data_dir):
    pos_dir = os.path.join(data_dir, 'pos')
    neg_dir = os.path.join(data_dir, 'neg')
    pos_fns = [os.path.join(pos_dir, fn) for fn in os.listdir(pos_dir)]
    neg_fns = [os.path.join(neg_dir, fn) for fn in os.listdir(neg_dir)]
    pos_data = [open(fn, 'r').read() for fn in pos_fns]
    neg_data = [open(fn, 'r').read() for fn in neg_fns]
    return [(p, 1) for p in pos_data] + [(n, 0) for n in neg_data]


def build_data(data, dev_ratio: float = 0.1, test_ratio: float = 0.1):
    train_eval_sep = int(len(data) * dev_ratio)
    eval_test_sep = int(len(data) * (dev_ratio + test_ratio))
    random.shuffle(data)
    train_data, eval_data, test_data = data[:train_eval_sep], data[train_eval_sep:eval_test_sep], data[eval_test_sep:]
    return train_data, eval_data, test_data