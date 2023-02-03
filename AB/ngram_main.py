import os
import json

from utils import build_dataset
from models.ngram import *

if __name__ == '__main__':
    train_dataset, eval_dataset, test_dataset = build_dataset('wikitext')
    output_dir = 'output'
    results = {}
    unigram_model = UnigramModel()
    unigram_model.train(train_dataset)
    results['unigram'] = {
        'train': unigram_model.perplexity(train_dataset),
        'eval': unigram_model.perplexity(eval_dataset)
    }
    results['bigram'], results['trigram'] = {}, {}
    for i in range(11):
        eps = 1 * (10 ** (-i))
        bigram_model, trigram_model = BigramModel(eps=eps), TrigramModel(eps=eps)
        bigram_model.train(train_dataset)
        trigram_model.train(train_dataset)
        results['bigram'][i] = {
        'train': bigram_model.perplexity(train_dataset),
        'eval': bigram_model.perplexity(eval_dataset)
        }
        results['trigram'][i] = {
        'train': trigram_model.perplexity(train_dataset),
        'eval': trigram_model.perplexity(eval_dataset)
        }
    json.dump(results, open(os.path.join(output_dir, 'ngram_wikitext_results.json'), 'w'))