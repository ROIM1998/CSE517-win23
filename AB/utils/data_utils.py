import os
import numpy as np
import random
from typing import List
from collections import Counter

_vocab = []
_vocab_cnt = {}
_vocab_to_id = {}
oovs = set()

def read_file(file_path, to_lower: bool = False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    if to_lower:
        lines = [line.lower() for line in lines]
    return lines

def reset_vocab():
    global _vocab, _vocab_cnt, _vocab_to_id, oovs
    _vocab = []
    _vocab_cnt = {}
    _vocab_to_id = {}
    oovs = set()

def text_to_feature(lines: List[str], cutoff: int = 3, unk_token: str = '[UNK]', stop_token: str = '[STOP]', start_token: str = '[CLS]', pad_stop: bool = True, pad_start: bool = True):
    global _vocab_cnt, _vocab_to_id, _vocab, oovs
    if not _vocab:
        _vocab_cnt = dict(Counter([w for l in lines for w in l.split()]))
        if unk_token not in _vocab_cnt:
            _vocab.append(unk_token)
            _vocab_to_id[unk_token] = len(_vocab_to_id)
        if pad_stop:
            _vocab.append(stop_token)
            _vocab_to_id[stop_token] = len(_vocab_to_id)
        if pad_start:
            _vocab.append(start_token)
            _vocab_to_id[start_token] = len(_vocab_to_id)
        for k, v in _vocab_cnt.items():
            if v >= cutoff:
                _vocab.append(k)
                _vocab_to_id[k] = len(_vocab_to_id)
            else:
                oovs.add(k)
    
    features = []
    for line in lines:
        features += ([_vocab_to_id[start_token]] if pad_start else []) + [_vocab_to_id[w] if w in _vocab_to_id else _vocab_to_id[unk_token] for w in line.split()] + ([_vocab_to_id[stop_token]] if pad_stop else [])  
        
    return np.array(features)

def load_glove_model(file_dir):
    glove_model = {}
    with open(file_dir,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    return glove_model

# Assuming negative label is 0 and positive is 1
def get_raw_data(data_dir):
    pos_dir = os.path.join(data_dir, 'pos')
    neg_dir = os.path.join(data_dir, 'neg')
    pos_fns = [os.path.join(pos_dir, fn) for fn in os.listdir(pos_dir)]
    neg_fns = [os.path.join(neg_dir, fn) for fn in os.listdir(neg_dir)]
    pos_data = [open(fn, 'r').read() for fn in pos_fns]
    neg_data = [open(fn, 'r').read() for fn in neg_fns]
    return [(p, 1) for p in pos_data] + [(n, 0) for n in neg_data]


def build_data_split(data, dev_ratio: float = 0.1, test_ratio: float = 0.1):
    dev_test_sep = int(len(data) * dev_ratio)
    test_train_sep = int(len(data) * (dev_ratio + test_ratio))
    random.shuffle(data)
    eval_data, test_data, train_data = data[:dev_test_sep], data[dev_test_sep:test_train_sep], data[test_train_sep:]
    return train_data, eval_data, test_data