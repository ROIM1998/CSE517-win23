import os
import random
import string
import numpy as np

from collections import Counter
from tqdm import tqdm

_vocab_cnt = {}
_vocab_to_id = {}
_vocab = []

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
    dev_test_sep = int(len(data) * dev_ratio)
    test_train_sep = int(len(data) * (dev_ratio + test_ratio))
    random.shuffle(data)
    eval_data, test_data, train_data = data[:dev_test_sep], data[dev_test_sep:test_train_sep], data[test_train_sep:]
    return train_data, eval_data, test_data

def preprocess_text(text: str):
    # Removing punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Converting all words to lower cases
    text = text.lower()
    # Converting newline into spaces, and separate by space for tokenization
    text = text.replace('\n', ' ').split()
    return text

def text_to_feature(text: str):
    text = preprocess_text(text)
    for word in set(text):
        if word not in _vocab_cnt:
            _vocab_cnt[word] = 1
            _vocab_to_id[word] = len(_vocab_to_id)
            _vocab.append(word)
        else:
            _vocab_cnt[word] += 1
    return dict(Counter(text))

def data_to_feature(data, cutoff=50):
    global _vocab_cnt, _vocab_to_id, _vocab
    # Extracting TF_IDF feature from raw texts
    data_with_cnt = [(text_to_feature(x), y) for x, y in tqdm(data)]
    _vocab = []
    _vocab_to_id = {}
    new_vocab_cnt = {}
    oov_cnt = 0
    for word, cnt in _vocab_cnt.items():
        if cnt >= cutoff:
            _vocab.append(word)
            _vocab_to_id[word] = len(_vocab_to_id)
            new_vocab_cnt[word] = cnt
        else:
            oov_cnt += cnt
    _vocab_cnt = new_vocab_cnt
    data_with_tfidf = [(np.array([(x[word] if word in x else 0) * np.log(len(data_with_cnt) / _vocab_cnt[word]) for word in _vocab]), y) for x, y in tqdm(data_with_cnt)]
    return data_with_tfidf, _vocab, _vocab_cnt