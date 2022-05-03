import torch
from torch.utils.data import Dataset

class Text8Dataset(Dataset):
    def __init__(self, path, seq_len=256):
        self.path = path
        self.seq_len = seq_len
        with open(path, 'r') as f:
            self.data = f.read()[:16*32]

        self.stoi = {
    " ": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26
}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.data = [self.stoi[c] for c in self.data]
        self.data = torch.tensor(self.data)
        self.data = self.data.view(-1, self.seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    dataset = Text8Dataset('../text8', seq_len=16)
    print(dataset.vocab_size)
    print(dataset.data.shape)
    print(dataset.data[0])
    print(''.join([dataset.itos[i.item()] for i in dataset.data[0]]))