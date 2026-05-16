import os

import string
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import confusion_matrix

from src.data_processing.dataset import get_dataloaders
from src.models.models_cnn import SimpleCNN
from src.prediction.predict import get_class_names

def get_label_map(dataset_name):
    if dataset_name == "emnist_letters":
        return {i: chr(ord('A') + i) for i in range(26)}

    elif dataset_name == "mnist":
        return {i: str(i) for i in range(10)}

    elif dataset_name == "emnist_balanced":
        chars = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        return {i: chars[i] for i in range(len(chars))}

    elif dataset_name == "emnist_byclass":
        chars = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        return {i: chars[i] for i in range(len(chars))}

    else:
        return None
      

# Load the model
def load_model(model_path, device, num_classes):
    model = SimpleCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(
        checkpoint, dict
    ) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Get predictions and true labels from the dataloader
def get_predictions(model, loader, device):
    all_preds = []
    all_labels = []
    misclassified = []

    with torch.no_grad():
        sample_index = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append({
                        "index": sample_index + i,
                        "pred": preds[i].item(),
                        "true": labels[i].item(),
                        "image": images[i].cpu()
                    })

            sample_index += len(labels)

    return all_preds, all_labels, misclassified

# Compute accuracy
def compute_accuracy(preds, labels):
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / len(labels)

def format_label(class_index, class_names):
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return str(class_index)


# Plot confusion matrix
def plot_confusion_matrix(
    labels,
    preds,
    dataset_name,
    save_path="outputs/confusion_matrix.png"
):
    os.makedirs("outputs", exist_ok=True)

    cm = confusion_matrix(labels, preds)

    label_map = get_label_map(dataset_name)

    if label_map:
        class_names = [label_map[i] for i in range(len(label_map))]
    else:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Character")
    plt.ylabel("True Character")
    plt.title(f"CNN Confusion Matrix on {dataset_name.upper()}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


# Save misclassified examples
def save_misclassified_examples(
    misclassified,
    dataset_name,
    save_path="outputs/misclassified_examples.png",
    max_examples=9
):
    if not misclassified:
        print("No misclassified examples found.")
        return

    label_map = get_label_map(dataset_name)

    os.makedirs("outputs", exist_ok=True)

    examples = misclassified[:max_examples]

    cols = 3
    rows = (len(examples) + cols - 1) // cols

    plt.figure(figsize=(10, 10))

    for i, example in enumerate(examples):
        image = example["image"].squeeze(0) * 0.5 + 0.5
        true_label = format_label(example["true"], class_names)
        pred_label = format_label(example["pred"], class_names)

        plt.subplot(rows, cols, i + 1)

        plt.imshow(example["image"].squeeze(0), cmap="gray")

        true_label = label_map[example["true"]]
        pred_label = label_map[example["pred"]]

        plt.title(f"True: {true_label}\nPred: {pred_label}")

        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved misclassified examples to {save_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN checkpoint.")
    parser.add_argument(
        "--dataset",
        default="emnist_byclass",
        choices=["mnist", "emnist", "emnist_letters", "emnist_byclass"],
        help="Dataset split the checkpoint was trained on.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Checkpoint path. Defaults to models/cnn_<dataset>.pth.",
    )
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()

# main evaluation function
def main():
    dataset_name = "emnist_byclass"
    batch_size = 64
    image_size = 28
    data_dir = "data"
    model_path = f"models/cnn_{dataset_name}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    print(f"Evaluating CNN on {dataset_name.upper()} with {num_classes} classes")
    print(f"Model checkpoint: {model_path}")

    model = load_model(model_path, device, num_classes=num_classes)
    class_names = get_class_names(dataset_name)

    preds, labels, misclassified = get_predictions(model, test_loader, device)

    accuracy = compute_accuracy(preds, labels)
    print(f"CNN test accuracy on {dataset_name.upper()}: {accuracy:.4f}")

    plot_confusion_matrix(labels, preds, dataset_name=dataset_name)
    save_misclassified_examples(misclassified, dataset_name=dataset_name)

    print("\nSample misclassifications:")
    for item in misclassified[:10]:
        true_label = format_label(item["true"], class_names)
        pred_label = format_label(item["pred"], class_names)
        print(f"Index {item['index']}: true={true_label}, pred={pred_label}")


if __name__ == "__main__":
    main()
