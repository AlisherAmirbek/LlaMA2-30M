from torch.utils.data import Dataset

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