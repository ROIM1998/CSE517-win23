import torch
import numpy as np

from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset
from .data_utils import read_file, text_to_feature, _vocab

def _remove_none(data):
    return [d['text'] for d in data if d['text']]

def build_dataset(data_source: str = 'wikitext') -> Tuple[Dataset]:
    if data_source == 'wikitext':
        unk_token, stop_token = '<unk>', '<stop>'
        dataset = load_dataset('wikitext', 'wikitext-2-v1')
        train_data, eval_data, test_data = dataset['train'], dataset['validation'], dataset['test']
        train_data, eval_data, test_data = _remove_none(train_data), _remove_none(eval_data), _remove_none(test_data)
    train_dataset = text_to_feature(train_data, unk_token=unk_token, stop_token=stop_token)
    eval_dataset, test_dataset = text_to_feature(eval_data, unk_token=unk_token, stop_token=stop_token), text_to_feature(test_data, unk_token=unk_token, stop_token=stop_token)
    return train_dataset, eval_dataset, test_dataset


def build_dataloader(data: np.array, batch_size: int = 32) -> torch.LongTensor:
    data = torch.LongTensor(data)
    num_batches = len(data) // batch_size 
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data