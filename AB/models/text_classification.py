import math
import torch
import torch.nn as nn

from utils.vocab_utils import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GloveTextClassification(nn.Module):
    def __init__(self, glove_vocab: Vocab, len_vocab: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.glove = glove_vocab
        self.glove_vocab_len = self.glove.glove_vocab_len
        self.embedding_dim = len(self.glove.get_weight('the'))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.glove_embedding = nn.Embedding(self.glove_vocab_len, self.embedding_dim)
        self.glove_embedding.weight = nn.Parameter(torch.from_numpy(self.glove.glove_embeddings).float())
        self.trained_embedding = nn.Embedding(len_vocab - self.glove_vocab_len, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, input_ids, text_lengths):
        mask = input_ids >= self.glove_vocab_len
        glove_batch = input_ids.clone()
        glove_batch[mask] = 0
        glove_embed = self.glove_embedding(glove_batch)
        glove_embed[mask] = 0
        trained_batch = input_ids.clone() - self.glove_vocab_len
        trained_batch[~mask] = 0
        trained_embed = self.trained_embedding(trained_batch)
        trained_embed[~mask] = 0
        embed = glove_embed + trained_embed
        embed = self.dropout(embed)
        packed_embedded = pack_padded_sequence(embed, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output[torch.arange(len(output)),text_lengths - 1,:])
        output = self.relu(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

    def init_weights(self):
        initrange_emb = 0.1
        initrange_other = 1 / math.sqrt(self.hidden_dim)
        self.trained_embedding.weight.data.uniform_(-initrange_emb, initrange_emb)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange_other, initrange_other)
        for i in range(self.num_layers):
            self.rnn.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                    self.hidden_dim).uniform_(-initrange_other, initrange_other) 
            self.rnn.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, 
                    self.hidden_dim).uniform_(-initrange_other, initrange_other)