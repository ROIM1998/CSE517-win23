import os
import torch
import json
import argparse
import time
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_tsv, data_split

args = argparse.ArgumentParser()
args.add_argument('--source_lang', type=str, default='norwegian')
args.add_argument('--target_lang', type=str, default='hindi')
args.add_argument('--model_name', type=str, default='/data0/zbw/models/mt5-small')
args.add_argument('--num_epochs', type=int, default=5)
args.add_argument('--eval_steps', type=int, default=200)
args = args.parse_args()

LANGTOCOL = {'chinese': 0, 'english': 1, 'spanish': 2, 'hindi': 3, 'japanese': 4, 'norwegian': 5}
COLTOLANG = {0: 'chinese', 1: 'english', 2: 'spanish', 3: 'hindi', 4: 'japanese', 5: 'norwegian'}

class TranslationDataset(Dataset):
    def __init__(self, inputs, outputs=None):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs['input_ids'])
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs['input_ids'][idx])
        attention_mask = torch.tensor(self.inputs['attention_mask'][idx])
        if self.outputs is None:
            return input_ids, attention_mask
        decoder_labels = torch.tensor(self.outputs['input_ids'][idx])
        decoder_labels[decoder_labels == tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, decoder_labels

def get_data(train_df, eval_df, tokenizer, source_lang, target_lang):
    # Tokenize and prepare dataset for training
    train_inputs = tokenizer(train_df[LANGTOCOL[source_lang.lower()]].tolist(), padding=True, max_length=32)
    train_outputs = tokenizer(train_df[LANGTOCOL[target_lang.lower()]].tolist(), padding=True, max_length=32)

    eval_inputs = tokenizer(eval_df[LANGTOCOL[source_lang.lower()]].tolist(), padding=True, max_length=32)
    eval_outputs = tokenizer(eval_df[LANGTOCOL[target_lang.lower()]].tolist(), padding=True, max_length=32)

    train_dataset = TranslationDataset(train_inputs, train_outputs)
    eval_dataset = TranslationDataset(eval_inputs, eval_outputs)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    return train_loader, eval_loader

def add_instruction(df):
    new_data = []
    prefix_template = 'translate {} to {}: '
    for i in tqdm(range(len(df))):
        for src in COLTOLANG:
            for tgt in COLTOLANG:
                if src != tgt:
                    prefix = prefix_template.format(COLTOLANG[src], COLTOLANG[tgt])
                    new_data.append([prefix + df.iloc[i][src], df.iloc[i][tgt]])
    return pd.DataFrame(new_data)

def evaluate(model, tokenizer, eval_loader):
    model.eval()
    eval_loss = 0
    accurate, total = 0, 0
    start_time = time.time()
    for batch in tqdm(eval_loader):
        input_ids, attention_mask, decoder_labels = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_labels)
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
        for batch in tqdm(train_loader):
            steps += 1
            input_ids, attention_mask, decoder_labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_labels).loss
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


def inference(model, tokenizer, device, test_dataloader):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=32)
            prediction = [tokenizer.decode(v) for v in outputs.tolist()]
            prediction = [v[:v.find(tokenizer.eos_token)] for v in prediction]
            prediction = [v.replace('<pad> ', '') for v in prediction]
            predictions.extend(prediction)
    return predictions

def fill_answer(test_df, model, tokenizer, device):
    filled_df = test_df.copy()
    test_dataset = []
    targets = []
    for i in tqdm(range(len(test_df))):
        row = test_df.iloc[i]
        source_id = [i for i in range(6) if isinstance(row[i], str) and row[i] != '?'][0]
        target_id = [i for i in range(6) if row[i] == '?'][0]
        source_lang, target_lang = COLTOLANG[source_id], COLTOLANG[target_id]
        prefix = 'translate {} to {}: '.format(source_lang, target_lang)
        test_dataset.append(prefix + row[source_id])
        targets.append(target_id)
    test_inputs = tokenizer(test_dataset, padding=True, max_length=32)
    test_dataset = TranslationDataset(test_inputs)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    predictions = inference(model, tokenizer, device, test_dataloader)
    for i, pred in enumerate(predictions):
        filled_df.iloc[i, targets[i]] = pred
    return filled_df

if __name__ == '__main__':
    source_lang = args.source_lang
    model_name = args.model_name
    # Load English-Chinese translation dataset
    df = read_tsv('data/Train.tsv')
    df.loc[0, 5] = 'null'
    train_df, eval_df = data_split(df, train_ratio=0.8 if source_lang != 'all' else 0.95)
    if source_lang == 'all':
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        train_df, eval_df = add_instruction(train_df), add_instruction(eval_df)
        train_dataloader, eval_dataloader = get_data(train_df, eval_df, tokenizer, 'chinese', 'english')
        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_epochs=args.num_epochs
        train_loss, eval_loss, train_state = train(model, tokenizer, optimizer, train_dataloader, eval_dataloader, num_epochs=num_epochs, eval_steps=args.eval_steps)
        eval_results = evaluate(model, tokenizer, eval_dataloader)
        output_dir = os.path.join('outputs', 'all-in-one')
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
    else:
        for target_lang in LANGTOCOL:
            if target_lang == source_lang:
                continue
            print('source_lang: %s, target_lang: %s, model_name: %s' % (source_lang, target_lang, model_name))
            model = MT5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
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

    test_df = read_tsv('data/Test.tsv')
    filled_df = fill_answer(test_df, model, tokenizer, device)
    filled_df.to_csv('data/Test-output.tsv', sep='\t', index=False, header=False)