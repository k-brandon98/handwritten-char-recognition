import os
import matplotlib.pyplot as plt
from src.dataset import load_mnist

def show_samples():
    train_dataset, _ = load_mnist()
    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(9):
        image, label = train_dataset[i]
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/sample_mnist.png")
    plt.show()

if __name__ == "__main__":
    show_samples()