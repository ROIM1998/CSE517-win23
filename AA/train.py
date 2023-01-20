import json
import sys
import numpy as np
from models.model import LogisticRegression, LexiconModel
from utils.data_utils import get_raw_data, text_to_feature, build_data, data_to_feature, load_lexicon, analyze_model_diff
from utils.tokenizer import Tokenizer
from tqdm import tqdm


def test(model, dataset):
    model.eval()
    losses = []
    acc = []
    all_outputs = []
    for inputs, labels in dataset:
        loss, outputs = model.forward(inputs, labels)
        outputs = np.where(outputs > 0.5, 1, 0)
        all_outputs.append(outputs.tolist())
        accuracy = np.mean(outputs == labels)
        losses.append(loss)
        acc.append(accuracy)
    return {
        'loss': np.mean(losses),
        'accuracy': np.mean(acc)
    }, all_outputs

if __name__ == '__main__':
    if len(sys.argv) == 1:
        cutoff, lr = 20, 0.1
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
                model.train()
            training_step += 1
    print(f'Best accuracy: {best_acc}')
    json.dump({
        'best_epoch': best_epoch,
        'best_accuracy': best_acc,
        'eval_loss': best_eval_loss,
        'training_loss': train_metrics['loss'],
    }, open('results/logistic_regression-cutoff%d-lr%f.json' % (cutoff, lr), 'w'), indent=4, sort_keys=True)

    negatives, positives = load_lexicon('lexicon')
    l_model = LexiconModel(negatives, positives)
    l_data = [(text_to_feature(d[0]), d[1]) for d in eval_data]
    prediction = [l_model.predict(text) for text, label in l_data]
    labels = np.array([label for text, label in l_data])
    is_positive = np.array([int(v > 0) for v in prediction])
    accuracy = np.mean(is_positive == labels)
    print(f'Lexicon accuracy: {accuracy}')
    
    lexicon_pos_logistic_neg_select, lexicon_neg_logistic_pos_select, both_neg_select = analyze_model_diff(labels, is_positive, best_output, eval_data)