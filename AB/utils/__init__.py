import torch
import numpy as np

from typing import Tuple, List
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .data_utils import read_file, text_to_feature, _vocab

def _remove_none(data):
    return [d['text'] for d in data if d['text']]

def build_dataset(data_source: str = 'wikitext') -> Tuple[Dataset]:
    if data_source == 'wikitext':
        unk_token, stop_token = '<unk>', '<stop>'
        dataset = load_dataset('wikitext', 'wikitext-2-v1')
        train_data, eval_data, test_data = dataset['train'], dataset['validation'], dataset['test']
        train_data, eval_data, test_data = _remove_none(train_data), _remove_none(eval_data), _remove_none(test_data)
    elif data_source == '1b_benchmark':
        unk_token, stop_token = '[UNK]', '[STOP]'
        train_data, eval_data, test_data = read_file('data/1b_benchmark/1b_benchmark.train.tokens'), read_file('data/1b_benchmark/1b_benchmark.dev.tokens'), read_file('data/1b_benchmark/1b_benchmark.test.tokens')
    train_dataset = text_to_feature(train_data, unk_token=unk_token, stop_token=stop_token, pad_start=False)
    eval_dataset, test_dataset = text_to_feature(eval_data, unk_token=unk_token, stop_token=stop_token, pad_start=False), text_to_feature(test_data, unk_token=unk_token, stop_token=stop_token, pad_start=False)
    return train_dataset, eval_dataset, test_dataset


def build_dataloader(data: np.array, batch_size: int = 32) -> torch.LongTensor:
    data = torch.LongTensor(data)
    num_batches = len(data) // batch_size 
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data



def build_text_classification_dataloader(dataset: List[Tuple[List[int], int]], batch_size: int = 64, pad_token_id: int = 0):
    def collate_fn(batch):
        data = [torch.LongTensor(item[0]) for item in batch]
        text_lengths = torch.LongTensor([len(item[0]) for item in batch])
        label = torch.Tensor([[item[1]] for item in batch]).float()
        return pad_sequence(data, batch_first=True, padding_value=pad_token_id), text_lengths, label
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)