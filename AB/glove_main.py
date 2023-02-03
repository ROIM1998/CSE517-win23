import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from utils import build_text_classification_dataloader
from utils.data_utils import get_raw_data, build_data_split
from sklearn.neighbors import NearestNeighbors
from utils.vocab_utils import Vocab, Tokenizer
from models.text_classification import GloveTextClassification

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
        
def _prepare_inputs(input_ids, text_lengths, labels):
    input_ids = input_ids.to('cuda')
    text_lengths = text_lengths.to('cuda')
    labels = labels.to('cuda')
    return input_ids, text_lengths, labels

if __name__ == '__main__':
    # vocab = Vocab(glove_path='data/glove/glove.6B.300d.txt')
    # run_similarity_exp(vocab)
    # run_analogy_exp(vocab)
    
    # Start text classification experiment
    # Data preparation
    data = get_raw_data('/home/zbw/projects/CSE517-win23/AA/data/txt_sentoken')
    train_data, eval_data, _ = build_data_split(data, dev_ratio=0.2)
    corpus = ' '.join([w for d in train_data for w in d[0].split()])
    
    tokenizer = Tokenizer(
        corpus=corpus,
        glove_path='data/glove/glove.6B.300d.txt',
        unk_token='<unk>',
        pad_token='<pad>',
    )
    train_dataset = [(tokenizer.tokenize(d[0]), d[1]) for d in train_data]
    eval_dataset = [(tokenizer.tokenize(d[0]), d[1]) for d in eval_data]
    train_dataloader = build_text_classification_dataloader(train_dataset, batch_size=32, pad_token_id = tokenizer.pad_token_id)
    eval_dataloader = build_text_classification_dataloader(eval_dataset, batch_size=32, pad_token_id = tokenizer.pad_token_id)
    
    # Start training
    hidden_dim=512
    num_layers=2
    dropout=0.65
    lr = 1e-3
    epochs=100
    
    model = GloveTextClassification(tokenizer.vocab, len(tokenizer.vocab), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    for n, p in model.named_parameters():
        if 'glove_embedding' in n:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to('cuda')
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader, desc='Training: ', leave=False)):
            input_ids, text_lengths, labels = _prepare_inputs(*batch)
            outputs = model(input_ids, text_lengths)
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print("Epoch %d, training loss: %.4f" % (i, epoch_loss))
        model.eval()
        epoch_eval_loss = 0
        correctness, total = 0, 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(eval_dataloader, desc='Evaluating: ', leave=False)):
                input_ids, text_lengths, labels = _prepare_inputs(*batch)
                outputs = model(input_ids, text_lengths)
                loss = F.binary_cross_entropy(outputs, labels)
                epoch_eval_loss += loss.item()
                correctness += ((outputs > 0.5) == labels).sum().item()
                total += len(labels)
            epoch_eval_loss /= len(eval_dataloader)
        print("Epoch %d, dev loss: %.4f, accuracy: %.4f" % (i, epoch_eval_loss, correctness / total))