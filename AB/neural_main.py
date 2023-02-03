import os
import json
import torch
import time
import torch.optim as optim

from utils import build_dataset, build_dataloader, _vocab
from tqdm import tqdm
from models.neural import *
from models.ngram import *
def get_batch(source: torch.Tensor, i: int, seq_len: int):
    seq_len = min(seq_len, source.shape[1] - 1 - i)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]
    return data, target

def train(model, data, optimizer, loss_fn, batch_size: int, seq_len: int, clip: float):
    epoch_loss = 0
    model.train()
    num_batches = data.shape[-1]
    hidden = model.init_hidden(batch_size)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ', leave=False):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        src, target = get_batch(data, idx, seq_len)
        src, target = src.to(model.device), target.to(model.device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)

        prediction = prediction.reshape(batch_size * seq_len, -1)   
        target = target.reshape(-1)
        loss = loss_fn(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def evaluate(model, data, loss_fn, batch_size, seq_len):
    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, idx, seq_len)
            src, target = src.to(model.device), target.to(model.device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = loss_fn(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

if __name__ == '__main__':
    output_dir = 'output'
    
    # Model arguments (hyper-parameters)
    hidden_dim=1024
    embedding_dim=1024
    num_layers=2
    dropout_rate=.65
    tie_weights=True
    
    # Training arguments (hyper-parameters)
    n_epochs = 40
    seq_len=64
    clip=0.25
    batch_size=128
    lr = 1e-3

    
    
    train_dataset, eval_dataset, test_dataset = build_dataset(data_source='wikitext')
    train_dataloader, eval_dataloader, test_dataloader = build_dataloader(train_dataset, batch_size=batch_size), build_dataloader(eval_dataset, batch_size=batch_size), build_dataloader(test_dataset, batch_size=batch_size)
    num_batches = train_dataloader.shape[-1]
    train_dataloader = train_dataloader[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = eval_dataloader.shape[-1]
    eval_dataloader = eval_dataloader[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = test_dataloader.shape[-1]
    test_dataloader = test_dataloader[:, :num_batches - (num_batches -1) % seq_len]
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = LSTMModel(vocab_size=len(_vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout_rate, device=device, tie_weights=tie_weights)
    print("Model parameter count: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    
    log_history = []
    best_valid_loss = float('inf')
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = train(model, train_dataloader, optimizer, loss_fn, 
                    batch_size, seq_len, clip)
        valid_loss = evaluate(model, eval_dataloader, loss_fn, batch_size, 
                    seq_len)
        
        lr_scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best-val-lstm_lm.pt'))

        train_perplexity, valid_perplexity = math.exp(train_loss), math.exp(valid_loss)
        print(f'\tTrain Perplexity: {train_perplexity:.3f}')
        print(f'\tValid Perplexity: {valid_perplexity:.3f}')
        log_history.append({'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss, 'lr': optimizer.param_groups[0]['lr'], 'train_perplexity': train_perplexity, 'valid_perplexity': valid_perplexity})
    training_time = time.time() - start_time
    test_loss = evaluate(model, test_dataloader, loss_fn, batch_size, seq_len)
    json.dump(log_history, open(os.path.join(output_dir, 'train_states.json'), 'w'), indent=4)
    
    # Model comparison
    lambda_1, labmda_2, labmda_3 = 0.2, 0.7, 0.1
    e_model = EnsembleModel(lambda_1, labmda_2, labmda_3, len(_vocab))
    ngram_start_time = time.time()
    e_model.train(train_dataset)
    ngram_training_time = time.time() - ngram_start_time
    ngram_train_perplexity = e_model.perplexity(train_dataset)
    ngram_test_perplexity = e_model.perplexity(test_dataset)
    
    unigram_model = UnigramModel(len(_vocab))
    unigram_start_time = time.time()
    unigram_model.train(train_dataset)
    unigram_training_time = time.time() - unigram_start_time
    unigram_train_perplexity = unigram_model.perplexity(train_dataset)
    unigram_test_perplexity = unigram_model.perplexity(test_dataset)
    
    bigram_model = BigramModel(len(_vocab))
    bigram_start_time = time.time()
    bigram_model.train(train_dataset)
    bigram_training_time = time.time() - bigram_start_time
    bigram_train_perplexity = bigram_model.perplexity(train_dataset)
    bigram_test_perplexity = bigram_model.perplexity(test_dataset)
    
    trigram_model = TrigramModel(len(_vocab))
    trigram_start_time = time.time()
    trigram_model.train(train_dataset)
    trigram_training_time = time.time() - trigram_start_time
    trigram_train_perplexity = trigram_model.perplexity(train_dataset)
    trigram_test_perplexity = trigram_model.perplexity(test_dataset)