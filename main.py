from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from model import CRNN
from utils import transform, text_to_labels, VOCAB

print("📥 Loading dataset...")
dataset = load_dataset("Teklia/IAM-line", split="train")

def collate_fn(batch):
    images = [transform(x["image"]) for x in batch]
    labels = [torch.tensor(text_to_labels(x["text"])) for x in batch]
    image_tensors = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return image_tensors, labels_padded, label_lengths

print("🔄 Preparing data loader...")
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# ✅ Optional: Overfit on 1 batch to debug
# Uncomment to test model learning capability
# loader = [next(iter(loader))]  # Use only one batch repeatedly

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")
model = CRNN(img_height=32, n_channels=1, n_classes=len(VOCAB) + 1).to(device)

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Increased LR for faster feedback

for epoch in range(5):
    model.train()
    total_loss, valid_batches = 0, 0

    for batch_idx, (images, labels, label_lengths) in enumerate(loader):
        images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)

        # ✅ Debug print: actual text and labels
        for i in range(min(3, len(labels))):
            print(f"🖼️ Sample {i+1}: Text: {dataset[batch_idx * 16 + i]['text']}")
            print(f"📚 Labels: {labels[i].tolist()} (len={label_lengths[i].item()})")

        logits = model(images)  # [B, T, C]
        log_probs = logits.log_softmax(2).permute(1, 0, 2)  # [T, B, C]
        input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long).to(device)

        # ⚠️ Skip if label longer than input sequence
        if (label_lengths >= input_lengths[0]).any():
            print(f"⚠️  Skipping batch {batch_idx+1} — input too short. Max label len: {label_lengths.max().item()}, input len: {input_lengths[0].item()}")
            continue

        loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Skipping batch {batch_idx+1} due to invalid loss: {loss.item()}")
            continue

        optimizer.zero_grad()
        loss.backward()

        # ✅ Debug: print gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"📈 Gradient norm: {total_norm:.4f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

        print(f"📦 Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(valid_batches, 1)
    print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "crnn_iam.pth")
