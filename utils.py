# utils.py
import string
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import string

# Define the vocabulary
VOCAB = string.ascii_letters + string.digits + string.punctuation + " "

# ✅ Unified charset used everywhere
CHARSET = string.ascii_letters + string.digits + string.punctuation + " "
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

# ✅ Common transform for preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor()
])

# ✅ Decode tensor to string using CTC rules
def decode(preds, vocab):
    chars = []
    prev = None
    for p in preds:
        if p != prev and p != 0:
            chars.append(vocab[p - 1])  # CTC blank is 0
        prev = p
    return ''.join(chars)

# ✅ Custom Dataset class for IAM
class IAMDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(CHARSET)}  # 0 is blank

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item['image'])
        text = item['text']
        label = torch.tensor([self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long)
        return image, label

# ✅ Collate function for CTC
def collate_fn(batch):
    images, labels = zip(*batch)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    images = torch.stack(images)
    return images, labels, label_lengths
