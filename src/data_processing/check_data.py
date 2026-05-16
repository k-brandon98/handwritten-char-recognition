import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.data_processing.dataset import load_dataset, create_splits

MNIST_CLASSES = list(range(10))


def _get_labels(dataset):
    if hasattr(dataset, 'indices'):  # torch Subset
        return dataset.dataset.targets[dataset.indices].tolist()
    return dataset.targets.tolist()


def show_samples(data_dir="data", save_path="outputs/sample_mnist.png"):
    train_dataset, _, _ = load_dataset(dataset_name="mnist", data_dir=data_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(9):
        image, label = train_dataset[i]
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_batch(dataloader, n=16, save_path="outputs/batch_sample.png"):
    images, labels = next(iter(dataloader))
    n = min(n, len(images))
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].squeeze().numpy(), cmap="gray")
        axes[i].set_title(str(labels[i].item()), fontsize=8)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Batch of {n} samples", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def plot_label_distribution(train_subset, val_subset, test_dataset,
                             save_path="outputs/label_distribution.png"):
    splits = [
        ("Train", _get_labels(train_subset)),
        ("Validation", _get_labels(val_subset)),
        ("Test", _get_labels(test_dataset)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, labels) in zip(axes, splits):
        counts = [labels.count(c) for c in MNIST_CLASSES]
        ax.bar(MNIST_CLASSES, counts)
        ax.set_title(f"{name} ({len(labels):,} samples)")
        ax.set_xlabel("Digit class")
        ax.set_ylabel("Count")
        ax.set_xticks(MNIST_CLASSES)

    plt.suptitle("Label Distribution per Split", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def plot_preprocessing_comparison(n=5, data_dir="data", save_path="outputs/preprocessing_comparison.png"):
    raw_dataset = datasets.MNIST(root=data_dir, train=True, download=True,
                                 transform=transforms.ToTensor())
    norm_dataset = datasets.MNIST(root=data_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

    for i in range(n):
        raw_img, label = raw_dataset[i]
        norm_img, _ = norm_dataset[i]

        axes[0, i].imshow(raw_img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Label: {label}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(norm_img.squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[1, i].set_title(f"min={norm_img.min():.2f}, max={norm_img.max():.2f}", fontsize=7)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Raw [0, 1]", fontsize=9)
    axes[1, 0].set_ylabel("Normalized [-1, 1]", fontsize=9)

    plt.suptitle("Preprocessing: Before vs After Normalization", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    full_train, test_dataset, _ = load_dataset(dataset_name="mnist")
    train_subset, val_subset = create_splits(full_train)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

    show_samples()
    plot_batch(train_loader)
    plot_label_distribution(train_subset, val_subset, test_dataset)
    plot_preprocessing_comparison()
