import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load trained model
from train_mnist import model  # Make sure 'model' is defined in train.py and trained

model.eval()

# Path to your image
image_path = "/img/digit.png"

# Image preprocessing (match MNIST)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),  # Ensure it's 1 channel
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

# Load and transform image
img = Image.open(image_path)
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).item()

# Show result
print(f"🧠 Predicted digit: {pred}")
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {pred}")
plt.axis('off')
plt.show()
