import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, n_channels, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(128 * (img_height // 4), 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.reshape(b, w, c * h)  # (B, W, C*H)
        x, _ = self.rnn(x)  # (B, W, hidden*2)
        x = self.fc(x)  # (B, W, n_classes)
        return x
