import numpy as np

from .data_utils import preprocess_text
from collections import Counter


class Tokenizer:
    def __init__(self, vocab, vocab_cnt, n):
        self.vocab = vocab
        self.vocab_cnt = vocab_cnt
        self.n = n
        
    def encode(self, text):
        text = preprocess_text(text)
        word_cnt = dict(Counter(text))
        tfidf = np.array([(word_cnt[word] if word in word_cnt else 0) * np.log(self.n / self.vocab_cnt[word]) for word in self.vocab])
        return tfidf