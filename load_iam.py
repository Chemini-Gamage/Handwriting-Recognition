from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("Teklia/IAM-line", split="train", streaming=True)
#  the first item from the stream is printed by next(iter(dataset) 
sample = next(iter(dataset))

# Display image and label
plt.imshow(sample["image"], cmap="gray")
plt.title(sample["text"])
plt.axis("off")
plt.show()
