import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, text, tokenizer, block_size):

        self.tokenizer = tokenizer
        self.data = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long
        )

        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):

        x = self.data[
            index:index + self.block_size
        ]

        y = self.data[
            index + 1:index + self.block_size + 1
        ]

        return x, y