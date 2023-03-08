import pandas as pd

def read_tsv(fn):
    return pd.read_csv(fn, sep='\t', header=None)


def data_split(df, train_ratio=0.8):
    df = df.sample(frac=1)
    train_df, eval_df = df[:int(len(df) * train_ratio)], df[int(len(df) * train_ratio):]
    return train_df, eval_df

def gen_eval_data(df):
    eval_data = []
    for i in range(len(df)):
        eval_data.append((df[0][i], df[1][i]))
    return eval_data

def sentence_to_tokens(sentence, use_split=True, retain_comma=True, retain_hyphen=False):
    if retain_comma:
        sentence = sentence.replace(',', ' ,')
    if retain_hyphen:
        sentence = sentence.replace('-', ' - ')
    if use_split:
        sentence = sentence.split()
    return [token.lower() for token in sentence]



# 0: chinese, 1: english, 2: spanish, 3: hindi, 4: japanese, 5: norwegian
class Tokenizer:
    def __init__(self, df=None):
        self.vocab = []
        self.vocab_to_idx = {}
        for v in df[0].tolist() + df[4].tolist():
            for token in sentence_to_tokens(v, use_split=False, retain_comma=False):
                if token not in self.vocab_to_idx:
                    self.vocab.append(token)
                    self.vocab_to_idx[token] = len(self.vocab_to_idx)

        for v in df[1].tolist():
            for token in sentence_to_tokens(v, use_split=True, retain_comma=False, retain_hyphen=True):
                if token not in self.vocab_to_idx:
                    self.vocab.append(token)
                    self.vocab_to_idx[token] = len(self.vocab_to_idx)

        for v in df[2].tolist() + df[3].tolist():
            for token in sentence_to_tokens(v, use_split=True, retain_comma=False):
                if token not in self.vocab_to_idx:
                    self.vocab.append(token)
                    self.vocab_to_idx[token] = len(self.vocab_to_idx)
        
        for v in df[5].tolist():
            for token in sentence_to_tokens(v, use_split=True, retain_comma=True):
                if token not in self.vocab_to_idx:
                    self.vocab.append(token)
                    self.vocab_to_idx[token] = len(self.vocab_to_idx)

    
    def tokenize(self, tokens):
        return [self.vocab_to_idx[token] for token in tokens]