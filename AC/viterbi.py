import os
import numpy as np

from tqdm import tqdm
from typing import List

DATA_DIR = 'data/viterbi'

def load_prob(fn='lm.txt'):
    with open(os.path.join(DATA_DIR, fn)) as f:
        lines = f.read().splitlines()
    word_pair, prob = zip(*[line.split('\t') for line in lines])
    probs = {}
    for words, p in zip(word_pair, prob):
        w1, w2 = words.split()
        probs[(w1, w2)] = np.log(float(p))
    return probs

probs = load_prob()
vocab = list(set([k[0] for k in probs.keys()]))

def load_corpus(fn='15pctmasked.txt'):
    with open(os.path.join(DATA_DIR, fn)) as f:
        lines = f.read().splitlines()
    return [v.split() for v in lines]

def optimize(prev_word: str, next_word: str, probs: dict) -> str:
    return max([(probs[(prev_word, w)] + probs[(w, next_word)], w) for w in vocab], key=lambda x: x[0])[1]

def viterbi(x: List[str]) -> List[str]:
    y = []
    score_mat = []
    backpoints = []
    
    def backtrack(w):
        reversed_tokens = []
        last_best = np.argmax([pre_score + probs[(vocab[pre_i], w)] for pre_i, pre_score in enumerate(score_mat[-1])])
        reversed_tokens.append(vocab[last_best])
        while len(backpoints) > 0:
            last_best = backpoints.pop()[last_best]
            reversed_tokens.append(vocab[last_best])
        return reversed_tokens[::-1]
    
    for i, w in enumerate(x):
        if w != '<mask>':
            if backpoints:
                y.extend(backtrack(w))
            score_mat = []
            backpoints = []
            y.append(w)
        elif x[i-1] != '<mask>' and x[i+1] != '<mask>':
            score_mat = []
            backpoints = []
            y.append(optimize(x[i-1], x[i+1], probs))
        elif x[i-1] != '<mask>':
            score_mat.append([probs[(x[i-1], w)] for w in vocab])
        else:
            new_score = []
            new_backpoints = []
            for new_w in vocab:
                possible_scores = [(score_mat[-1][pre_i] + probs[(pre_w, new_w)], pre_i) for pre_i, pre_w in enumerate(vocab)]
                max_score = max(possible_scores, key=lambda x: x[0])
                new_score.append(max_score[0])
                new_backpoints.append(max_score[1])
            score_mat.append(new_score)
            backpoints.append(new_backpoints)
    return y

def readable(w):
    w = ['' if v == '<start>' or v == '<eos>' else ' ' if v == '<s>' else v for v in w]
    return ''.join(w)

if __name__ == '__main__':
    data = load_corpus()
    unmasked_data = [viterbi(x) for x in tqdm(data)]
    with open(os.path.join(DATA_DIR, 'decoded.txt'), 'w') as f:
        f.write('\n'.join([' '.join(x) for x in unmasked_data]))