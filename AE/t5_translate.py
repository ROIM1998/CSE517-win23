import os
import torch
import json
import argparse
import time

from transformers import AutoTokenizer, MT5ForConditionalGeneration
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_tsv, data_split

args = argparse.ArgumentParser()
args.add_argument('--source_lang', type=str, default='english')
args.add_argument('--target_lang', type=str, default='hindi')
args.add_argument('--model_name', type=str, default='google/mt5-small')
args.add_argument('--num_epochs', type=int, default=5)
args.add_argument('--eval_steps', type=int, default=200)
args = args.parse_args()

LANGTOCOL = {'chinese': 0, 'english': 1, 'spanish': 2, 'hindi': 3, 'japanese': 4, 'norwegian': 5}

class TranslationDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs['input_ids'])
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs['input_ids'][idx])
        attention_mask = torch.tensor(self.inputs['attention_mask'][idx])
        decoder_input_ids = torch.tensor(self.outputs['input_ids'][idx])
        decoder_attention_mask = torch.tensor(self.outputs['attention_mask'][idx])
        decoder_labels = torch.tensor(self.outputs['input_ids'][idx])
        decoder_labels[decoder_labels == tokenizer.pad_token_id] = -100
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_labels

def get_data(train_df, eval_df, tokenizer, source_lang, target_lang):
    # Tokenize and prepare dataset for training
    train_inputs = tokenizer(train_df[LANGTOCOL[source_lang.lower()]].tolist(), padding=True, max_length=32)
    train_outputs = tokenizer(train_df[LANGTOCOL[target_lang.lower()]].tolist(), padding=True, max_length=32)

    eval_inputs = tokenizer(eval_df[LANGTOCOL[source_lang.lower()]].tolist(), padding=True, max_length=32)
    eval_outputs = tokenizer(eval_df[LANGTOCOL[target_lang.lower()]].tolist(), padding=True, max_length=32)

    train_dataset = TranslationDataset(train_inputs, train_outputs)
    eval_dataset = TranslationDataset(eval_inputs, eval_outputs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    return train_loader, eval_loader

def evaluate(model, tokenizer, eval_loader):
    model.eval()
    eval_loss = 0
    accurate, total = 0, 0
    start_time = time.time()
    for batch in eval_loader:
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_labels = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_labels)
            loss = outputs.loss
            total += input_ids.shape[0]
            predictions = [tokenizer.decode(v) for v in outputs.logits.argmax(dim=-1).tolist()]
            predictions = [v[:v.find(tokenizer.eos_token)] for v in predictions]
            labels = [v[:v.index(1)] for v in decoder_labels.tolist()]
            labels = [tokenizer.decode(v) for v in labels]
            accurate += sum([pred == l for pred, l in zip(predictions, labels)])
        eval_loss += loss.item()
    torch.cuda.synchronize()
    use_time = time.time() - start_time
    eval_loss /= len(eval_loader)
    eval_results = {'eval_loss': eval_loss, 'eval_accuracy': accurate/total, 'eval_time': use_time}
    return eval_results

def train(model, tokenizer, optimizer, train_loader, eval_loader, num_epochs=10, eval_steps=200):
    steps = 0
    train_state = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            steps += 1
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_labels).loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if steps % eval_steps == 0:
                eval_results = evaluate(model, tokenizer, eval_loader)
                eval_loss, eval_acc = eval_results['eval_loss'], eval_results['eval_accuracy']
                model.train()

                print(f'Epoch {steps/len(train_loader):.2f} - Train Loss: {train_loss/steps:.4f} - eval Loss: {eval_loss:.4f} - eval Acc: {eval_acc:.4f}')
                train_state.append({
                    'epoch': steps/len(train_loader),
                    'train_loss': train_loss/steps,
                    'eval_loss': eval_loss,
                    'eval_acc': eval_acc,
                })
    return train_loss / steps, eval_loss, train_state

if __name__ == '__main__':
    source_lang = args.source_lang
    target_lang = args.target_lang
    model_name = args.model_name
    print('source_lang: %s, target_lang: %s, model_name: %s' % (source_lang, target_lang, model_name))
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load English-Chinese translation dataset
    df = read_tsv('data/Train.tsv')
    train_df, eval_df = data_split(df)
    train_dataloader, eval_dataloader = get_data(train_df, eval_df, tokenizer, source_lang, target_lang)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs=args.num_epochs
    train_loss, eval_loss, train_state = train(model, tokenizer, optimizer, train_dataloader, eval_dataloader, num_epochs=num_epochs, eval_steps=args.eval_steps)
    eval_results = evaluate(model, tokenizer, eval_dataloader)
    output_dir = os.path.join('outputs', '%s-%s' % (source_lang, target_lang))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_loss,
    }, os.path.join(output_dir, 'model.pt'))

    json.dump(train_state, open(os.path.join(output_dir, 'train_state.json'), 'w'), indent=4, sort_keys=True)
    json.dump(eval_results, open(os.path.join(output_dir, 'eval_results.json'), 'w'), indent=4, sort_keys=True)