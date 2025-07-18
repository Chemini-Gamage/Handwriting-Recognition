# utils.py
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as TF
from PIL import Image
import string

# Charset includes lowercase, uppercase, digits, punctuation and space
charset = string.ascii_letters + string.digits + string.punctuation + " "
char2idx = {char: idx + 1 for idx, char in enumerate(charset)}  # 0 is reserved for blank
idx2char = {idx: char for char, idx in char2idx.items()}

def text_to_labels(text):
    return [char2idx[char] for char in text if char in char2idx]

def labels_to_text(labels):
    return ''.join([idx2char[idx] for idx in labels if idx in idx2char])

class IAMDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        label = item['text']

        if self.transform:
            img = self.transform(img)

        label_tensor = torch.tensor(text_to_labels(label), dtype=torch.long)
        return img, label_tensor, len(label_tensor)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths
