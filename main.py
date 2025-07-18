from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from model import CRNN
from utils import transform, text_to_labels, VOCAB

# Load dataset
dataset = load_dataset("Teklia/IAM-line", split="train")

def collate_fn(batch):
    images = [transform(x["image"]) for x in batch]
    labels = [torch.tensor(text_to_labels(x["text"])) for x in batch]
    image_tensors = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return image_tensors, labels_padded, label_lengths

loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=32, n_channels=1, n_classes=len(VOCAB) + 1).to(device)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels, label_lengths in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)  # (B, T, C)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
        input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)

        loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "crnn_iam.pth")
