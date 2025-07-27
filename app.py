import streamlit as st
from PIL import Image
import torch
from model import CRNN
from utils import transform, decode, VOCAB

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=32, n_channels=1, n_classes=len(VOCAB) + 1).to(device)
model.load_state_dict(torch.load("crnn_iam.pth", map_location=device))
model.eval()

st.title("✍️ Handwriting Recognition (CRNN)")
st.markdown("Upload a grayscale or color handwritten text image (preferably a single line).")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)  # (B, T, C)
        log_probs = logits.log_softmax(2)
        _, preds = log_probs.max(2)
        preds = preds[0].cpu().numpy()

    text = decode(preds.tolist(), VOCAB)  # ✅ Pass VOCAB explicitly
    st.success(f"📝 **Predicted Text:** {text}")
