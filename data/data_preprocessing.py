import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import re
import numpy as np
from datasets import load_dataset

path_to_tokenizer = ''
tokenizer = SentencePieceProcessor(path_to_tokenizer)

dataset_path = ""

shard_files = [
    f"{dataset_path}/data-00000-of-00080.arrow",
    f"{dataset_path}/data-00001-of-00080.arrow",
    f"{dataset_path}/data-00002-of-00080.arrow",
    f"{dataset_path}/data-00003-of-00080.arrow",
    f"{dataset_path}/data-00004-of-00080.arrow",
    f"{dataset_path}/data-00005-of-00080.arrow",
    f"{dataset_path}/data-00006-of-00080.arrow",
]

dataset = load_dataset('arrow', data_files=shard_files)

dataset = [re.sub(r'\n+', ' ', text) for text in tqdm(dataset['train']['text'])]

dataset = [text.replace("\"", '') for text in tqdm(dataset)]

dataset = [re.sub(r'\s+', ' ', text) for text in tqdm(dataset)]

from functools import lru_cache

@lru_cache(maxsize=None)
def cached_encode(char):
    return tokenizer.encode(char)

tokens = []
for text in tqdm(dataset):
    for char in text:
        token = cached_encode(char)
        if 4 in token:
            tokens.append(char)

tokens_set = set(str(char) for char in tokens)

def contains_token(text):
    return not tokens_set.isdisjoint(text)

filtered_texts = [text for text in tqdm(dataset) if not contains_token(text)]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = []

        current_chunk = []
        for text in tqdm(texts):
            # Tokenize the text
            encoded_text = tokenizer.encode(text, add_bos=True, add_eos = True)
            for token in encoded_text:
                current_chunk.append(token)
                if len(current_chunk) == max_length:
                    self.chunks.append(current_chunk)
                    current_chunk = []
        # Add the last chunk if it's not empty and not equal to max_length
        if current_chunk:
            self.chunks.append(current_chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self.chunks)}")
        chunk = self.chunks[idx]
        return torch.tensor(chunk, dtype=torch.long)

max_length = 513

dataset_tokenized = TextDataset(filtered_texts, tokenizer, max_length)

path_to_save = '/dataset_tokenized.pth'
torch.save(dataset_tokenized, path_to_save)