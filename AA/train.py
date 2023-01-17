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
    data = get_raw_data('data/txt_sentoken')
    train_data, eval_data, test_data = build_data(data)
    train_dataset, vocab, vocab_cnt = data_to_feature(train_data, cutoff=20)
    tokenizer = Tokenizer(vocab, vocab_cnt, len(train_dataset))
    print("Dimension of the vocabulary:", len(vocab))
    eval_dataset = [(tokenizer.encode(text), label) for text, label in tqdm(eval_data)]
    model = LogisticRegression(len(vocab), lr=0.1)
    epochs = 3
    log_step = 200
    training_step = 0
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
                model.train()
            training_step += 1