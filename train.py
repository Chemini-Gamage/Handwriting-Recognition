# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from model import CRNN
from utils import IAMDataset, collate_fn, charset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load IAM line dataset
dataset = load_dataset("Teklia/IAM-line", split="train")
val_dataset = load_dataset("Teklia/IAM-line", split="validation")

# 2. Transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor()
])

# 3. Create dataset & dataloaders
train_ds = IAMDataset(dataset, transform)
val_ds = IAMDataset(val_dataset, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 4. Model
model = CRNN(img_height=32, num_classes=len(charset) + 1).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels, label_lengths in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # (T, N, C)
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)

        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 6. Save model
torch.save(model.state_dict(), "crnn_iam.pth")
print("✅ Model saved to crnn_iam.pth")
