# predict_custom_images.py
import torch
from PIL import Image
from model import CRNN
from utils import CHARSET, NUM_CLASSES, transform, decode

# Load model
model = CRNN(img_height=32, n_channels=1, n_classes=NUM_CLASSES)
model.load_state_dict(torch.load("crnn_iam.pth", map_location="cpu"))
model.eval()

# Load image
img_path = "img/image1.jpg"
image = Image.open(img_path).convert("L")
image = transform(image).unsqueeze(0)  # [1, 1, 32, 128]

# Predict
with torch.no_grad():
    logits = model(image)  # [B, T, C]
    logits = logits.log_softmax(2)
    preds = logits.argmax(2).squeeze(0)  # [T]
    text = decode(preds, CHARSET)
    print(f"📝 Predicted text: {text}")
