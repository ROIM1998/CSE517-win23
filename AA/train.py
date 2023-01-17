import json
import sys
import numpy as np
from models.model import LogisticRegression
from utils.data_utils import get_raw_data, build_data, data_to_feature
from utils.tokenizer import Tokenizer
from tqdm import tqdm

def test(model, dataset):
    model.eval()
    losses = []
    acc = []
    for inputs, labels in dataset:
        loss, outputs = model.forward(inputs, labels)
        outputs = np.where(outputs > 0.5, 1, 0)
        accuracy = np.mean(outputs == labels)
        losses.append(loss)
        acc.append(accuracy)
    return {
        'loss': np.mean(losses),
        'accuracy': np.mean(acc)
    }

if __name__ == '__main__':
    if len(sys.argv) == 1:
        cutoff, lr = 20, 0.1
    elif len(sys.argv) == 3:
        cutoff, lr = int(sys.argv[1]), float(sys.argv[2])
    else:
        raise Exception('Invalid arguments')
    data = get_raw_data('data/txt_sentoken')
    train_data, eval_data, test_data = build_data(data)
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
    for i in range(epochs):
        for inputs, labels in train_dataset:
            loss, outputs = model.forward(inputs, labels)
            model.optimize()
            model.zero_grad()
            if not training_step % log_step:
                print(f'Epoch {i}, Step {training_step}, Loss {loss}')
                train_metrics = test(model, train_dataset)
                print(f'Training Loss {train_metrics["loss"]}, train Accuracy {train_metrics["accuracy"]}')
                eval_metrics = test(model, eval_dataset)
                print(f'Eval Loss {eval_metrics["loss"]}, Eval Accuracy {eval_metrics["accuracy"]}')
                if eval_metrics['accuracy'] > best_acc:
                    best_acc = eval_metrics['accuracy']
                    best_epoch = i
                    best_eval_loss = eval_metrics['loss']
                model.train()
            training_step += 1
    print(f'Best accuracy: {best_acc}')
    json.dump({
        'best_epoch': best_epoch,
        'best_accuracy': best_acc,
        'eval_loss': best_eval_loss,
        'training_loss': train_metrics['loss'],
    }, open('results/logistic_regression-cutoff%d-lr%f.json' % (cutoff, lr), 'w'), indent=4, sort_keys=True)