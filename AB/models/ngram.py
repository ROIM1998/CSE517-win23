from typing import List

import numpy as np

class NgramModel:
    def __init__(self):
        pass
    
    def train(self, train_dataset):
        pass
    
    def predict_next_word(self, prefix):
        pass
    
    def evaluate(self, sentence):
        pass
    
    def perplexity(self, dataset):
        pass
    
    
class UnigramModel(NgramModel):
    def __init__(self):
        self.theta = None
        self.num_train_tokens = None
        
    def train(self, train_dataset):
        unique, counts = np.unique(np.concatenate(train_dataset), return_counts=True)
        self.num_train_tokens = np.sum(counts)
        self.theta = counts / self.num_train_tokens
        
    def predict_next_word(self, prefix):
        return self.theta.argmax()
    
    def evaluate(self, sentence, eps=1e-300):
        if isinstance(sentence, np.ndarray) and sentence.ndim == 1:
            return np.prod(self.theta[sentence]) + eps
        elif isinstance(sentence, list) or (isinstance(sentence, np.ndarray) and sentence.ndim == 2):
            return np.mean([np.prod(self.theta[s]) for s in sentence]) + eps
        else:
            return None
    
    def perplexity(self, dataset: List[np.ndarray]):
        ce_loss = np.mean([- np.log2(self.evaluate(s)) / len(s) for s in dataset])
        return 2 ** ce_loss
    
class BigramModel(NgramModel):
    def __init__(self, eps=1):
        self.theta = None
        self.bigram_cnt = {}
        self.eps = eps
    
    def train(self, train_dataset):
        num_unique_tokens = len(np.unique(np.concatenate(train_dataset)))
        for s in train_dataset:
            for i in range(len(s) - 1):
                if (s[i], s[i + 1]) not in self.bigram_cnt:
                    self.bigram_cnt[(s[i], s[i + 1])] = 1 + self.eps
                else:
                    self.bigram_cnt[(s[i], s[i + 1])] += 1

        self.theta = np.ones([num_unique_tokens, num_unique_tokens])
        for (w1, w2), cnt in self.bigram_cnt.items():
            self.theta[w1, w2] = cnt
        self.theta = (self.theta.T / self.theta.sum(axis=1)).T
                    
    def predict_next_word(self, prefix):
        return self.theta[prefix[-1]].argmin()
    
    def evaluate(self, sentence, eps=1e-300):
        if isinstance(sentence, np.ndarray) and sentence.ndim == 1:
            return np.prod([self.theta[sentence[i], sentence[i + 1]] for i in range(len(sentence) - 1)]) + eps
        elif isinstance(sentence, list) or (isinstance(sentence, np.ndarray) and sentence.ndim == 2):
            return np.mean([self.evaluate(s) for s in sentence]) + eps
        else:
            return None
        
    def perplexity(self, dataset: List[np.ndarray]):
        ce_loss = np.mean([- np.log2(self.evaluate(s)) / len(s) for s in dataset])
        return 2 ** ce_loss