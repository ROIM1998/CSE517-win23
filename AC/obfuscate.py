import os
import random
import pandas as pd

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