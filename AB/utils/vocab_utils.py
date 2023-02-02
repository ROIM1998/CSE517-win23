import numpy as np
from .data_utils import read_file, load_glove_model
from typing import List
from collections import Counter

class Vocab:
    def __init__(self, corpus_path: str = None, glove_path: str = None, cutoff_frequency: int = 3):
        self.corpus_path = corpus_path
        self.glove_path = glove_path
        self.glove_weights = None
        self._vocab = []
        self._vocab_cnt = {}
        self._vocab_to_id = {}
        self.glove_embeddings = None
        if glove_path:
            self.glove_weights = load_glove_model(glove_path)
            self._vocab_cnt = None
            self.glove_embeddings = []
            for k, v in self.glove_weights.items():
                self._vocab.append(k)
                self._vocab_to_id[k] = len(self._vocab_to_id)
                self.glove_embeddings.append(v)
            self.glove_embeddings = np.stack(self.glove_embeddings)
        if corpus_path:
            lines = read_file(corpus_path, to_lower=True)
            corpus_vocab_cnt = dict(Counter([w for l in lines for w in l.split()]))
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