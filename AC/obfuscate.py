import os
import random
import gensim
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from nltk import tokenize

DATA_DIR = 'data/nlp_ethic/'

def load_wordlist(fn):
    with open(fn) as f:
        words = f.read().splitlines()
    return set(words)

male_words = load_wordlist(os.path.join(DATA_DIR, 'male.txt'))
female_words = load_wordlist(os.path.join(DATA_DIR, 'female.txt'))

def word_replace(text: str, target: str) -> str:
    assert target in ['M', 'W']
    tokens = tokenize.word_tokenize(text.lower())
    if target == 'M':
        replace_func = lambda x: random.choice(list(male_words)) if x in female_words else x
    elif target == 'W':
        replace_func = lambda x: random.choice(list(female_words)) if x in male_words else x
    else:
        raise ValueError('Invalid target')
    return ' '.join([replace_func(token) for token in tokens])

def baseline_obfuscation(data_fn):
    df = pd.read_csv(os.path.join(DATA_DIR, data_fn))
    df.loc[df['op_gender'] == 'M','post_text'] = df[df['op_gender'] == 'M']['post_text'].apply(lambda x: word_replace(x, 'W'))
    df.loc[df['op_gender'] == 'W','post_text'] = df[df['op_gender'] == 'W']['post_text'].apply(lambda x: word_replace(x, 'M'))
    df.to_csv(os.path.join(DATA_DIR, 'baseline_obfuscated_' + data_fn), index=False)
    
def semantic_obfuscation(data_fn):
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(DATA_DIR, 'GoogleNews-vectors-negative300.bin'), binary=True)
    # Constructing word2emb matrices  
    female_emb = []
    female_word_list = []
    for w in female_words:
        if w in model:
            female_emb.append(model[w])
            female_word_list.append(w)
    
    male_emb = []
    male_word_list = []
    for w in male_words:
        if w in model:
            male_emb.append(model[w])
            male_word_list.append(w)
    reduced_female_wordset = set(female_word_list)
    reduced_male_wordset = set(male_word_list)
    
    # Building nearest-neighbor search structures
    female_nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine')
    female_nn.fit(np.array(female_emb))
    male_nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine')
    male_nn.fit(np.array(male_emb))
    
    def semantic_replace(text: str, target: str) -> str:
        assert target in ['M', 'W']
        tokens = tokenize.word_tokenize(text.lower())
        if target == 'M':
            replace_func = lambda x: male_word_list[male_nn.kneighbors(model[x].reshape(1, -1))[1].item()] if x in reduced_female_wordset else x
        elif target == 'W':
            replace_func = lambda x: female_word_list[female_nn.kneighbors(model[x].reshape(1, -1))[1].item()] if x in reduced_male_wordset else x
        else:
            raise ValueError('Invalid target')
        return ' '.join([replace_func(token) for token in tokens])
    
    df = pd.read_csv(os.path.join(DATA_DIR, data_fn))
    df.loc[df['op_gender'] == 'M','post_text'] = df[df['op_gender'] == 'M']['post_text'].apply(lambda x: semantic_replace(x, 'W'))
    df.loc[df['op_gender'] == 'W','post_text'] = df[df['op_gender'] == 'W']['post_text'].apply(lambda x: semantic_replace(x, 'M'))
    df.to_csv(os.path.join(DATA_DIR, 'semantic_obfuscated_' + data_fn), index=False)