import torch
from PIL import Image
from model import CRNN
from utils import transform, labels_to_text, VOCAB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=32, n_channels=1, n_classes=len(VOCAB) + 1).to(device)
model.load_state_dict(torch.load("crnn_iam.pth", map_location=device))
model.eval()

# Load your image (must be a handwritten line)
img = Image.open("img/image.jpg")  # ← put your test image here
img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 32, 128)

with torch.no_grad():
    output = model(img_tensor)
    probs = output.log_softmax(2)
    pred = probs.argmax(2)[0].cpu().numpy()
    decoded = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:
            decoded.append(p)
        prev = p

    print("🧠 Prediction:", labels_to_text(decoded))
