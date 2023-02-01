import torch
import torch.nn as nn
import math

from typing import Tuple
from torch.nn import LSTM

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float, tie_weights: bool = True, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, batch_first=True)
        self.dropout = nn.Dropout(self.dropout)
        self.cls = nn.Linear(self.hidden_dim, self.vocab_size)
        if tie_weights:
            assert embedding_dim == hidden_dim, "Tied weights can only be used when embedding_dim == hidden_dim"
            self.embedding.weight = self.cls.weight
        self.init_weights()
        self.device = device
        self.to(device)
        
    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor = None, output_hidden_states: bool = True):
        # input_ids: (batch_size, seq_len)
        # output: (batch_size, seq_len, vocab_size)
        output = self.dropout(self.embedding(input_ids))
        output, hidden = self.rnn(output, hidden)
        output = self.dropout(output)
        output = self.cls(output)
        if output_hidden_states:
            return output, hidden
        else:
            return output
        
    def init_weights(self):
        initrange_emb = 0.1
        initrange_other = 1 / math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-initrange_emb, initrange_emb)
        self.cls.bias.data.zero_()
        self.cls.weight.data.uniform_(-initrange_other, initrange_other)
        for i in range(self.num_layers):
            self.rnn.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                    self.hidden_dim).uniform_(-initrange_other, initrange_other) 
            self.rnn.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, 
                    self.hidden_dim).uniform_(-initrange_other, initrange_other)
                    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor]:
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        return (hidden, cell)
    
    def detach_hidden(self, hidden: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        hidden, cell = hidden
        return hidden.detach(), cell.detach()