import numpy as np
from .data_utils import read_file, load_glove_model
from typing import List, Optional
from collections import Counter

class Vocab:
    def __init__(self, corpus_path: str = None, corpus: str = None, glove_path: str = None, cutoff_frequency: int = 3, to_lower: bool = False, unk_token: Optional[str] = None, stop_token: Optional[str] = None, start_token: Optional[str] = None, pad_token: Optional[str] = None):
        self.corpus_path = corpus_path
        self.glove_path = glove_path
        self.glove_weights = None
        self._vocab = []
        self._vocab_to_id = {}
        self.glove_embeddings = None
        if corpus is not None and corpus_path is not None:
            raise ValueError('Only one of corpus or corpus_path can be specified')
        if corpus_path:
            lines = read_file(corpus_path, to_lower=to_lower)
            corpus = ' '.join(lines)
        corpus_vocab_cnt = {}
        if corpus:
            if to_lower:
                corpus = corpus.lower()
            corpus_vocab_cnt = dict(Counter(corpus.split()))
        else:
            raise ValueError('Either corpus or corpus_path must be specified')
        if glove_path:
            self.glove_weights = load_glove_model(glove_path)
            self.glove_embeddings = []
            for k, v in self.glove_weights.items():
                if k in corpus_vocab_cnt and corpus_vocab_cnt[k] >= cutoff_frequency:
                    self._vocab.append(k)
                    self._vocab_to_id[k] = len(self._vocab_to_id)
                    self.glove_embeddings.append(v)
            self.glove_embeddings = np.stack(self.glove_embeddings)
        self.glove_vocab_len = len(self._vocab)
        for token in [unk_token, stop_token, start_token, pad_token]:
            if token:
                self._vocab.append(token)
                self._vocab_to_id[token] = len(self._vocab_to_id)
        for k, v in corpus_vocab_cnt.items():
            if v >= cutoff_frequency and k not in self._vocab_to_id:
                self._vocab.append(k)
                self._vocab_to_id[k] = len(self._vocab_to_id)

    def __len__(self):
        return len(self._vocab)
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._vocab_to_id[idx]
        else:
            return self._vocab[idx]
        
    def get_weight(self, idx):
        if isinstance(idx, str):
            return self.glove_weights[idx]
        else:
            return self.glove_embeddings[idx]
        
class Tokenizer:
    def __init__(self, corpus_path: str = None, corpus: str = None, glove_path: str = None, cutoff_frequency: int = 3, to_lower: bool = False, unk_token: Optional[str] = None, stop_token: Optional[str] = None, start_token: Optional[str] = None, pad_token: Optional[str] = None, pad_stop: bool = False, pad_start: bool = False):
        self.vocab = Vocab(corpus_path, corpus, glove_path, cutoff_frequency, to_lower, unk_token, stop_token, start_token, pad_token)
        self.to_lower = to_lower
        self.unk_token = unk_token
        self.stop_token = stop_token
        self.start_token = start_token
        self.pad_token = pad_token
        if pad_token is not None:
            self.pad_token_id = self.vocab._vocab_to_id[pad_token]
        self.pad_stop = pad_stop
        self.pad_start = pad_start
        if pad_stop and not stop_token:
            raise ValueError("pad_stop is set to True but no stop_token is provided")
        if pad_start and not start_token:
            raise ValueError("pad_start is set to True but no start_token is provided")
        
    def tokenize(self, text: str):
        if self.to_lower:
            text = text.lower()
        return ([self.vocab._vocab_to_id[self.start_token]] if self.pad_start else []) + [self.vocab._vocab_to_id[w] if w in self.vocab._vocab_to_id else self.vocab._vocab_to_id[self.unk_token] for w in text.split()] + ([self.vocab._vocab_to_id[self.stop_token]] if self.pad_stop else [])
    
    def __call__(self, text: str):
        return self.tokenize(text)