import os
import numpy as np
from typing import List
from collections import Counter

_vocab = []
_vocab_cnt = {}
_vocab_to_id = {}
oovs = set()

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def text_to_feature(lines: List[str]):
    global _vocab_cnt, _vocab_to_id, _vocab, oovs
    _vocab_cnt = dict(Counter([w for l in lines for w in l.split()]))
    _vocab.append('[STOP]')
    _vocab_to_id['[STOP]'] = len(_vocab_to_id)
    _vocab.append('[UNK]')
    _vocab_to_id['[UNK]'] = len(_vocab_to_id)
    for k, v in _vocab_cnt.items():
        if v >= 3:
            _vocab.append(k)
            _vocab_to_id[k] = len(_vocab_to_id)
        else:
            oovs.add(k)
    features = [
        np.array([_vocab_to_id[w] if w not in oovs else _vocab_to_id['[UNK]'] for w in line.split()] + [_vocab_to_id['[STOP]']])
        for line in lines
    ]
    return features