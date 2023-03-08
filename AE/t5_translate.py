import torch

from transformers import AutoTokenizer, MT5ForConditionalGeneration
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_tsv, data_split

# Load pre-trained MT5 model and tokenizer
model_name = 'google/mt5-small'
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load English-Chinese translation dataset
df = read_tsv('data/Train.tsv')
train_df, eval_df = data_split(df)

# Tokenize and prepare dataset for training
train_inputs = tokenizer(train_df[0].tolist(), padding=True, max_length=32)
train_outputs = tokenizer(train_df[1], padding=True, max_length=32)

eval_inputs = tokenizer(eval_df[0], padding=True, max_length=32)
eval_outputs = tokenizer(eval_df[1], padding=True, max_length=32)


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

train_dataset = TranslationDataset(train_inputs, train_outputs)
eval_dataset = TranslationDataset(eval_inputs, eval_outputs)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_labels).loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    eval_loss = 0
    for batch in eval_loader:
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_labels = [b.to(device) for b in batch]
        with torch.no_grad():
            loss = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_labels).loss
        eval_loss += loss.item()
    eval_loss /= len(eval_loader)

print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f} - eval Loss: {eval_loss:.4f}')

