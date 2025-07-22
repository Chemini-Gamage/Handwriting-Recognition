# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import CRNN
from utils import IAMDataset, collate_fn, transform, CHARSET, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IAM dataset
dataset = load_dataset("Teklia/IAM-line", split="train")
val_dataset = load_dataset("Teklia/IAM-line", split="validation")

# Prepare data
train_ds = IAMDataset(dataset, transform)
val_ds = IAMDataset(val_dataset, transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize model
model = CRNN(img_height=32, n_channels=1, n_classes=NUM_CLASSES).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels, label_lengths in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # [B, T, C]
        outputs = outputs.permute(1, 0, 2)  # CTC expects [T, B, C]
        input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "crnn_iam.pth")
print("✅ Model saved as crnn_iam.pth")
