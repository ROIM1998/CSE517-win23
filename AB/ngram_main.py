import os
import json
import seaborn as sns
sns.set_theme(style="darkgrid")
import numpy as np

from utils import build_dataset
from utils.data_utils import _vocab, reset_vocab
from models.ngram import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    train_dataset, eval_dataset, test_dataset = build_dataset('1b_benchmark')
    output_dir = 'output'
    results = {}
    unigram_model = UnigramModel(vocab_len=len(_vocab))
    unigram_model.train(train_dataset)
    results['unigram'] = {
        'train': unigram_model.perplexity(train_dataset),
        'eval': unigram_model.perplexity(eval_dataset)
    }
    results['bigram'], results['trigram'] = {}, {}
    for i in range(11):
        eps = 1 * (10 ** (-i))
        bigram_model, trigram_model = BigramModel(vocab_len=len(_vocab), eps=eps), TrigramModel(vocab_len=len(_vocab), eps=eps)
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
    
    # Plotting epsilon-perplexity correlation
    sns.lineplot(x=[int(k) for k in results['bigram']], y=[np.log10(v['train']) for v in results['bigram'].values()], label='bigram_train')
    sns.lineplot(x=[int(k) for k in results['bigram']], y=[np.log10(v['eval']) for v in results['bigram'].values()], label='bigram_eval')
    sns.lineplot(x=[int(k) for k in results['trigram']], y=[np.log10(v['train']) for v in results['trigram'].values()], label='trigram_train')
    sns.lineplot(x=[int(k) for k in results['trigram']], y=[np.log10(v['eval']) for v in results['trigram'].values()], label='trigram_eval')

    plt.xlabel('-log10(eps)')
    plt.ylabel('log10(perplexity)')
    plt.legend()
    plt.savefig('ngram_perplexity.png')
    plt.clf()
    
    lambda_1, labmda_2, labmda_3 = 0.1, 0.3, 0.6
    model = EnsembleModel(lambda_1, labmda_2, labmda_3)
    model.train(train_dataset)
    model.perplexity(train_dataset)
    model.perplexity(eval_dataset)
    
    results = []
    lambda_1, labmda_2, labmda_3 = 0.2, 0.7, 0.1
    training_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for ratio in training_ratio:
        model = EnsembleModel(lambda_1, labmda_2, labmda_3, len(_vocab))
        model.train(train_dataset[:int(ratio * len(train_dataset))])
        results.append({
            'train': model.perplexity(train_dataset[:int(ratio * len(train_dataset))]),
            'eval': model.perplexity(eval_dataset),
            'test': model.perplexity(test_dataset),
            'ratio': ratio,
        })
        print(f'Eval perplexity for ratio {ratio} is {model.perplexity(test_dataset)}, with training perplexity {model.perplexity(train_dataset[:int(ratio * len(train_dataset))])}')
    sns.lineplot(x=[r['ratio'] for r in results], y=[np.log10(r['train']) for r in results], label='train')
    sns.lineplot(x=[r['ratio'] for r in results], y=[np.log10(r['eval']) for r in results], label='eval')
    # sns.lineplot(x=[r['ratio'] for r in results], y=[np.log10(r['test']) for r in results], label='test')
    plt.xlabel('Training ratio')
    plt.ylabel('log10(perplexity)')
    plt.legend()
    plt.savefig('training_ratio_perplexity.png')
    plt.clf()
    
    results = []
    for cutoff in [15, 20, 25, 30, 35, 40, 45, 50]:
        reset_vocab()
        train_dataset, eval_dataset, test_dataset = build_dataset('1b_benchmark', cutoff=cutoff)
        from utils.data_utils import _vocab
        model = EnsembleModel(lambda_1, labmda_2, labmda_3, len(_vocab))
        model.train(train_dataset)
        results.append({
            'train': model.perplexity(train_dataset),
            'eval': model.perplexity(eval_dataset),
            'test': model.perplexity(test_dataset),
            'cutoff': cutoff,
        })
        print(f'Eval perplexity for cutoff {cutoff} is {model.perplexity(test_dataset)}, with training perplexity {model.perplexity(train_dataset)}')
        
    sns.lineplot(x=[r['cutoff'] for r in results], y=[np.log10(r['train']) for r in results], label='train')
    sns.lineplot(x=[r['cutoff'] for r in results], y=[np.log10(r['eval']) for r in results], label='eval')
    # sns.lineplot(x=[r['ratio'] for r in results], y=[np.log10(r['test']) for r in results], label='test')
    plt.xlabel('Token frequency cutoff')
    plt.ylabel('log10(perplexity)')
    plt.legend()
    plt.savefig('cutoff_perplexity.png')
    plt.clf()