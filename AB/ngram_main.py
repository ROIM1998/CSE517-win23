import json

from utils.data_utils import read_file, text_to_feature
from models.ngram import *

if __name__ == '__main__':
    train_path, eval_path = 'AB/data/1b_benchmark/1b_benchmark.train.tokens', 'AB/data/1b_benchmark/1b_benchmark.dev.tokens'
    train_raw_data, eval_raw_data = read_file(train_path), read_file(eval_path)
    train_dataset = text_to_feature(train_raw_data)
    eval_dataset = text_to_feature(eval_raw_data)
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
    json.dump(results, open('ngram_results.json', 'w'))