# predict_custom_images.py

import torch
from PIL import Image
from model import CRNN
from utils import CHARSET, NUM_CLASSES, transform, decode

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CRNN(img_height=32, n_channels=1, n_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("crnn_iam.pth", map_location=device))
model.eval()

# Load and preprocess image
img_path = "img/image4.jpg"
image = Image.open(img_path).convert("L")  # Convert to grayscale
input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 1, 32, W]

# Predict
with torch.no_grad():
    logits = model(input_tensor)  # Shape: [T, B, C]
    log_probs = logits.log_softmax(2)
    preds = log_probs.argmax(2).squeeze(1)  # Shape: [T]

# Decode prediction
text = decode(preds.squeeze().tolist(), CHARSET)

print(f"📝 Predicted text: {text}")
