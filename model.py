import torch
import torch.nn as nn
import string
from utils import VOCAB
class CRNN(nn.Module):
    def __init__(self, img_height=32, n_channels=1, n_classes=len(VOCAB) + 1):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),  # -> [B, 64, 32, W]
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> [B, 64, 16, W]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> [B, 128, 8, W]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> [B, 256, 4, W/2]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> [B, 512, 2, W/2]
        )

        # RNN layers
        self.rnn1 = nn.LSTM(input_size=512 * 2, hidden_size=256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)

        # Final classification layer
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 2, W/2]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, W/2, C, H]
        x = x.reshape(b, w, c * h)  # [B, W/2, 512*2]

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)

        x = self.fc(x)  # [B, W/2, n_classes]
        return x


# 🧪 Dummy test when running directly
if __name__ == "__main__":
    # Charset: alphanumerics + punctuation + space
    charset = string.ascii_letters + string.digits + string.punctuation + " "
    n_classes = len(charset) + 1  # 1 for CTC blank

    model = CRNN(img_height=32, n_channels=1, n_classes=n_classes)

    dummy_input = torch.randn(1, 1, 32, 128)  # [B, C, H, W]
    logits = model(dummy_input)

    print("✅ Logits shape:", logits.shape)  # [1, seq_len, n_classes]
    print("📊 Max class index:", torch.argmax(logits, dim=2).max().item())
    print("🔤 Number of classes (with blank):", n_classes)
