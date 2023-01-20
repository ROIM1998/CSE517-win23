import json
import sys
import numpy as np
from models.model import LogisticRegression, LexiconModel
from utils.data_utils import get_raw_data, text_to_feature, build_data, data_to_feature, load_lexicon, analyze_model_diff
from utils.tokenizer import Tokenizer
from tqdm import tqdm

def comb(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

def test(model, dataset):
    model.eval()
    losses = []
    acc = []
    all_outputs = []
    all_labels = []
    for inputs, labels in dataset:
        loss, outputs = model.forward(inputs, labels)
        outputs = np.where(outputs > 0.5, 1, 0)
        all_outputs.append(outputs.tolist())
        accuracy = np.mean(outputs == labels)
        losses.append(loss)
        acc.append(accuracy)
        all_labels.append(labels)
    precision = len([1 for i, j in zip(all_outputs, all_labels) if i == j and i == 1]) / len([1 for i in all_outputs if i == 1]) if len([1 for i in all_outputs if i == 1]) else 0
    recall = len([1 for i, j in zip(all_outputs, all_labels) if i == j and i == 1]) / len([1 for i in all_labels if i == 1])
    return {
        'loss': np.mean(losses),
        'accuracy': np.mean(acc),
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall),
    }, all_outputs

if __name__ == '__main__':
    if len(sys.argv) == 1:
        cutoff, lr = 50, 0.3
    elif len(sys.argv) == 3:
        cutoff, lr = int(sys.argv[1]), float(sys.argv[2])
    else:
        raise Exception('Invalid arguments')
    data = get_raw_data('data/txt_sentoken')
    train_data, eval_data, _ = build_data(data, dev_ratio=0.2)
    train_dataset, vocab, vocab_cnt = data_to_feature(train_data, cutoff=cutoff)
    tokenizer = Tokenizer(vocab, vocab_cnt, len(train_dataset))
    print("Dimension of the vocabulary:", len(vocab))
    eval_dataset = [(tokenizer.encode(text), label) for text, label in tqdm(eval_data)]
    model = LogisticRegression(len(vocab), lr=lr)
    epochs = 10
    log_step = 200
    training_step = 0
    best_acc = 0
    best_eval_loss = 0
    best_epoch = -1
    best_output = None
    best_metric = None
    for i in range(epochs):
        for inputs, labels in train_dataset:
            loss, outputs = model.forward(inputs, labels)
            model.optimize()
            model.zero_grad()
            if not training_step % log_step:
                print(f'Epoch {i}, Step {training_step}, Loss {loss}')
                train_metrics, _ = test(model, train_dataset)
                print(f'Training Loss {train_metrics["loss"]}, train Accuracy {train_metrics["accuracy"]}')
                eval_metrics, outputs = test(model, eval_dataset)
                print(f'Eval Loss {eval_metrics["loss"]}, Eval Accuracy {eval_metrics["accuracy"]}')
                if eval_metrics['accuracy'] > best_acc:
                    best_acc = eval_metrics['accuracy']
                    best_epoch = i
                    best_eval_loss = eval_metrics['loss']
                    best_output = outputs
                    best_metric = eval_metrics
                model.train()
            training_step += 1
    print(f'Best accuracy: {best_acc}')
    json.dump({
        'best_epoch': best_epoch,
        'best_accuracy': best_acc,
        'eval_loss': best_eval_loss,
        'training_loss': train_metrics['loss'],
        'precision': best_metric['precision'],
        'recall': best_metric['recall'],
        'f1': best_metric['f1'],
    }, open('results/logistic_regression-cutoff%d-lr%f.json' % (cutoff, lr), 'w'), indent=4, sort_keys=True)

    negatives, positives = load_lexicon('lexicon')
    l_model = LexiconModel(negatives, positives)
    l_data = [(text_to_feature(d[0]), d[1]) for d in eval_data]
    prediction = [l_model.predict(text) for text, label in l_data]
    labels = np.array([label for text, label in l_data])
    is_positive = np.array([int(v > 0) for v in prediction])
    l_precision = sum(labels & is_positive) / sum(is_positive)
    l_recall = sum(labels & is_positive) / sum(labels)
    l_f1 = 2 * l_precision * l_recall / (l_precision + l_recall)
    accuracy = np.mean(is_positive == labels)
    print(f'Lexicon accuracy: {accuracy}, precision: {l_precision}, recall: {l_recall}, f1: {l_f1}')
    
    # Significance test below
    alpha = 0.05
    logi_1_lexi_1, logi_1_lexi_0, logi_0_lexi_1, logi_0_lexi_0 = 0, 0, 0, 0
    for logi, lexi, l in zip(best_output, is_positive, labels):
        logi_correct, lexi_correct = logi == l, lexi == l
        if logi_correct and lexi_correct:
            logi_1_lexi_1 += 1
        elif logi_correct and not lexi_correct:
            logi_1_lexi_0 += 1
        elif not logi_correct and lexi_correct:
            logi_0_lexi_1 += 1
        elif not logi_correct and not lexi_correct:
            logi_0_lexi_0 += 1
    k = min(logi_1_lexi_0, logi_0_lexi_1)
    p = 1 / (2 ** (logi_1_lexi_0 + logi_0_lexi_1 - 1)) * sum([comb(logi_1_lexi_0 + logi_0_lexi_1, j) for j in range(k + 1)])
    if p < 0.05:
        print("Reject null hypothesis, the difference is significant")
    else:
        print("Accept null hypothesis, the difference is not significant")