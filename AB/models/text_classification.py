import torch
import torch.nn as nn

from utils.data_utils import load_glove_model

class GloveTextClassification(nn.Module):
    def __init__(self, glove_path: str, len_vocab: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.glove = load_glove_model(glove_path)
        self.glove_vocab_len = len(self.glove)
        self.embedding_dim = len(self.glove['the'])
        self.glove_embedding = nn.Embedding.from_pretrained(self.glove)
        self.trained_embedding = nn.Embedding(len_vocab - self.glove_vocab_len, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        mask = input_ids >= self.glove_vocab_len
        glove_batch = input_ids.clone()
        glove_batch[mask] = 0
        glove_embed = self.glove_embedding(glove_batch)
        trained_batch = input_ids.clone() - self.glove_vocab_len
        trained_batch[~mask] = 0
        trained_embed = self.trained_embedding(trained_batch)
        embed = glove_embed + trained_embed
        embed = self.dropout(embed)
        output, (hidden, cell) = self.rnn(embed)
        output = self.dropout(output)
        output = self.fc(output)
        return output