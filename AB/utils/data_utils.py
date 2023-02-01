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

def text_to_feature(lines: List[str], cutoff: int = 3, unk_token: str = '[UNK]', stop_token: str = '[STOP]', pad_stop: bool = True):
    global _vocab_cnt, _vocab_to_id, _vocab, oovs
    if not _vocab:
        _vocab_cnt = dict(Counter([w for l in lines for w in l.split()]))
        _vocab.append(stop_token)
        _vocab_to_id[stop_token] = len(_vocab_to_id)
        _vocab.append(unk_token)
        _vocab_to_id[unk_token] = len(_vocab_to_id)
        for k, v in _vocab_cnt.items():
            if v >= cutoff:
                _vocab.append(k)
                _vocab_to_id[k] = len(_vocab_to_id)
            else:
                oovs.add(k)
        features = [
            np.array([_vocab_to_id[w] if w not in oovs else _vocab_to_id[unk_token] for w in line.split()] + ([_vocab_to_id[stop_token]] if pad_stop else []))
            for line in lines
        ]
    else:
        features = [
            np.array([_vocab_to_id[w] if w in _vocab_to_id else _vocab_to_id[unk_token] for w in line.split()] + ([_vocab_to_id[stop_token]] if pad_stop else []))
            for line in lines
        ]
    return features