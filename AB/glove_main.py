import time
import os
import torch
import json
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from utils import build_text_classification_dataloader
from utils.data_utils import get_raw_data, build_data_split
from sklearn.neighbors import NearestNeighbors
from utils.vocab_utils import Vocab, Tokenizer
from models.text_classification import GloveTextClassification
from sklearn.metrics import f1_score

def run_similarity_exp(vocab: Vocab):
    model = NearestNeighbors(n_neighbors=2,
                            metric='cosine',
                            algorithm='brute',
                            n_jobs=-1)
    model.fit(vocab.glove_embeddings)
    
    # Word similarity experiment
    query_words = ['dog', 'whale', 'before', 'however', 'fabricate']
    X = np.stack([vocab.glove_embeddings[vocab._vocab_to_id[word]] for word in query_words])
    distances, indices = model.kneighbors(X)
    for i, idx in enumerate(indices):
        print("The closest word to word '%s' is '%s' with distance %.4f" % (query_words[i], vocab[idx[1]], distances[i][1]))
        
def run_analogy_exp(vocab: Vocab):
    # Word analogy experiment
    model = NearestNeighbors(n_neighbors=3,
                            metric='cosine',
                            algorithm='brute',
                            n_jobs=-1)
    model.fit(vocab.glove_embeddings)
    analogy_words = [
        ['dog', 'puppy', 'cat'],
        ['speak', 'speaker', 'sing'],
        ['france', 'french', 'england'],
        ['france', 'wine', 'england'],
    ]
    X = np.stack([vocab.get_weight(words[1]) + vocab.get_weight(words[2]) - vocab.get_weight(words[0]) for words in analogy_words])
    distances, indices = model.kneighbors(X)
    for i, idx in enumerate(indices):
        print("The closest word to word analogy combination '%s' is '%s' with distance %.4f" % (analogy_words[i], vocab[idx[0]], distances[i][0]))
        print("The second closest word to word analogy combination '%s' is '%s' with distance %.4f" % (analogy_words[i], vocab[idx[1]], distances[i][1]))
        print("The third closest word to word analogy combination '%s' is '%s' with distance %.4f" % (analogy_words[i], vocab[idx[2]], distances[i][2]))
        
def _prepare_inputs(input_ids, text_lengths, labels, device):
    input_ids = input_ids.to(device)
    text_lengths = text_lengths.to(device)
    labels = labels.to(device)
    return input_ids, text_lengths, labels

def evaluate(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    correctness, total = 0, 0
    eval_labels, eval_predictions = [], []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc='Evaluating: ', leave=False)):
            input_ids, text_lengths, labels = _prepare_inputs(*batch, device)
            logits = model(input_ids, text_lengths)
            loss = F.cross_entropy(logits, labels)
            epoch_loss += loss.item()
            prediction = (logits > 0.5).long()
            correctness += (prediction == labels).sum().item()
            total += len(labels)
            eval_labels.extend(labels.cpu().numpy().tolist())
            eval_predictions.extend(prediction.cpu().numpy().tolist())
    eval_f1 = f1_score(eval_labels, eval_predictions, average='macro')
    return epoch_loss / len(dataloader), correctness / total, eval_f1

if __name__ == '__main__':
    # vocab = Vocab(glove_path='data/glove/glove.6B.300d.txt')
    # run_similarity_exp(vocab)
    # run_analogy_exp(vocab)
    
    # Start text classification experiment
    # Data preparation
    data = get_raw_data('/home/zbw/projects/CSE517-win23/AA/data/txt_sentoken')
    train_data, eval_data, test_data = build_data_split(data, dev_ratio=0.2)
    corpus = ' '.join([w for d in train_data for w in d[0].split()])
    
    tokenizer = Tokenizer(
        corpus=corpus,
        glove_path='data/glove/glove.6B.300d.txt',
        unk_token='<unk>',
        pad_token='<pad>',
    )
    train_dataset = [(tokenizer.tokenize(d[0]), d[1]) for d in train_data]
    eval_dataset = [(tokenizer.tokenize(d[0]), d[1]) for d in eval_data]
    test_dataset = [(tokenizer.tokenize(d[0]), d[1]) for d in test_data]
    train_dataloader = build_text_classification_dataloader(train_dataset, batch_size=32, pad_token_id = tokenizer.pad_token_id)
    eval_dataloader = build_text_classification_dataloader(eval_dataset, batch_size=32, pad_token_id = tokenizer.pad_token_id)
    test_dataloader = build_text_classification_dataloader(test_dataset, batch_size=32, pad_token_id = tokenizer.pad_token_id)
    # Start training
    hidden_dim=512
    num_layers=2
    dropout=0.65
    lr = 1e-3
    epochs=100
    device = 'cuda'
    output_dir = 'output'
    
    model = GloveTextClassification(tokenizer.vocab, len(tokenizer.vocab), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    for n, p in model.named_parameters():
        if 'glove_embedding' in n:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    log_history = []
    best_eval_loss = float('inf')
    start_time = time.time()
    test_loss, test_acc, test_f1 = None, None, None
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader, desc='Training: ', leave=False)):
            input_ids, text_lengths, labels = _prepare_inputs(*batch, device=device)
            outputs = model(input_ids, text_lengths)
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print("Epoch %d, training loss: %.4f" % (i, epoch_loss))
        eval_loss, eval_acc, eval_f1 = evaluate(model, eval_dataloader, device)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'glove_textclf_model.pt'))
            test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, device)
        log_history.append({
            'epoch': i,
            'train_loss': epoch_loss,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_acc,
            'eval_f1': eval_f1,
        })
        print("Epoch %d, dev loss: %.4f, accuracy: %.4f, f1: %.4f" % (i, eval_loss, eval_acc, eval_f1))
    print("Training finished, total time: %.4f" % (time.time() - start_time))
    print("Test loss: %.4f, accuracy: %.4f, f1: %.4f" % (test_loss, test_acc, test_f1))
    json.dump(log_history, open(os.path.join(output_dir, 'glove_textclf_log.json'), 'w'), indent=4)