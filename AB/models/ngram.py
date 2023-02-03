from typing import List, Union

import numpy as np

class NgramModel:
    def __init__(self):
        pass
    
    def train(self, train_dataset):
        pass
    
    def predict_next_word(self, prefix):
        pass
    
    def evaluate(self, sentence):
        raise NotImplementedError
    
    def perplexity(self, dataset: Union[np.array, List[np.ndarray]]):
        if isinstance(dataset, list):
            dataset = np.concatenate(dataset)
        ce_loss = self.evaluate(dataset, return_log=True)
        return 2 ** ce_loss
    
    
class UnigramModel(NgramModel):
    def __init__(self, vocab_len: int, eps: float = 1.):
        self.theta = np.zeros(vocab_len) * eps
        
    def train(self, train_dataset):
        unique, counts = np.unique(train_dataset, return_counts=True)
        for i, cnt in zip(unique, counts):
            self.theta[i] += cnt
        self.theta /= np.sum(self.theta)
        
    def predict_next_word(self, prefix):
        return self.theta
    
    def evaluate(self, sentence, return_log: bool = True, return_raw: bool = False):
        probabilities = self.theta[sentence]
        if return_raw:
            return np.array(probabilities)
        if not return_log:
            return np.array(probabilities)
        else:
            return (- np.sum(np.log2(probabilities))) / len(sentence)
    
class BigramModel(NgramModel):
    def __init__(self, vocab_len: int, eps=1e-3):
        self.theta = np.ones([vocab_len, vocab_len]) * eps
        self.bigram_cnt = {}
        self.eps = eps
        self.inner_unigram = UnigramModel(vocab_len=vocab_len)
    
    def train(self, train_dataset):
        self.inner_unigram.train(train_dataset)
        for i in range(len(train_dataset) - 1):
            if (train_dataset[i], train_dataset[i + 1]) not in self.bigram_cnt:
                self.bigram_cnt[(train_dataset[i], train_dataset[i + 1])] = 1
            else:
                self.bigram_cnt[(train_dataset[i], train_dataset[i + 1])] += 1

        for (w1, w2), cnt in self.bigram_cnt.items():
            self.theta[w1, w2] += cnt
        self.theta = (self.theta.T / self.theta.sum(axis=1)).T
                    
    def predict_next_word(self, prefix):
        return self.theta[prefix[-1]]
    
    def evaluate(self, sentence, return_log: bool = True, return_raw: bool = False):
        probabilities = [self.inner_unigram.evaluate(sentence[0], return_raw=True).item()] + [self.theta[sentence[i], sentence[i + 1]] for i in range(len(sentence) - 1)]
        if return_raw:
            return np.array(probabilities)
        if not return_log:
            return np.prod(probabilities)
        else:
            return (- np.sum(np.log2(probabilities))) / len(sentence)
    
class TrigramModel(NgramModel):
    def __init__(self, vocab_len: int, eps=1e-2):
        self._vocab_len = 0
        self.trigram_cnt = {}
        self.prefix_cnt = {}
        self.eps = eps
        self.inner_bigram = BigramModel(vocab_len=vocab_len, eps=eps)
        
    def train(self, train_dataset):
        self.inner_bigram.train(train_dataset)
        num_unique_tokens = len(set(train_dataset))
        self._vocab_len = num_unique_tokens
        s = train_dataset
        for i in range(len(s) - 2):
            if (s[i], s[i + 1]) not in self.prefix_cnt:
                self.prefix_cnt[(s[i], s[i + 1])] = 1
            else:
                self.prefix_cnt[(s[i], s[i + 1])] += 1
            if (s[i], s[i + 1], s[i + 2]) not in self.trigram_cnt:
                self.trigram_cnt[(s[i], s[i + 1], s[i + 2])] = 1
            else:
                self.trigram_cnt[(s[i], s[i + 1], s[i + 2])] += 1
        for (w1, w2, w3), cnt in self.trigram_cnt.items():
            self.trigram_cnt[(w1, w2, w3)] = (cnt + self.eps) / (self.prefix_cnt[(w1, w2)] + self.eps * self._vocab_len)
        
                    
    def predict_next_word(self, prefix):
        return np.array([self.trigram_cnt[(prefix[-2], prefix[-1], i)] if (prefix[-2], prefix[-1], i) in self.trigram_cnt else self.eps / (self.prefix_cnt[(prefix[-2], prefix[-1])] + self.eps * self._vocab_len) if (prefix[-2], prefix[-1]) in self.prefix_cnt else self.eps / self._vocab_len for i in range(self._vocab_len)])
    
    def evaluate(self, sentence, return_log: bool = True, return_raw: bool = False):
        probabilities = self.inner_bigram.evaluate(sentence[:2], return_raw=True).tolist() + [self.trigram_cnt[(sentence[i], sentence[i + 1], sentence[i + 2])] if (sentence[i], sentence[i + 1], sentence[i + 2]) in self.trigram_cnt else self.eps / (self.prefix_cnt[(sentence[i], sentence[i + 1])] + self.eps * self._vocab_len) if (sentence[i], sentence[i + 1]) in self.prefix_cnt else self.eps / self._vocab_len for i in range(len(sentence) - 2)]
        if return_raw:
            return np.array(probabilities)
        if not return_log:
            return np.prod(probabilities)
        else:
            return (- np.sum(np.log2(probabilities))) / len(sentence)
        
        
class EnsembleModel(NgramModel):
    def __init__(self, lambda_1, lambda_2, lambda_3, vocab_len):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.unigram = UnigramModel(vocab_len=vocab_len)
        self.bigram = BigramModel(vocab_len=vocab_len, eps=1e-3)
        self.trigram = TrigramModel(vocab_len=vocab_len, eps=1e-2)
        
    def train(self, train_dataset):
        self.unigram.train(train_dataset)
        self.bigram.train(train_dataset)
        self.trigram.train(train_dataset)
    
    def evaluate(self, sentence, return_log: bool = True):
        probabilities = self.lambda_1 * self.unigram.evaluate(sentence, return_raw=True) + self.lambda_2 * self.bigram.evaluate(sentence, return_raw=True) + self.lambda_3 * self.trigram.evaluate(sentence, return_raw=True)
        if not return_log:
            return np.prod(probabilities)
        else:
            return (- np.sum(np.log2(probabilities))) / len(sentence)
        