import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from dataset import get_dataloaders
from model_baseline import BaselineLogisticRegression
from model_cnn import CharacterCNN
from sklearn.metrics import confusion_matrix


def load_model(model_type, model_path, device, image_size=28, num_classes=10):
    if model_type == "baseline":
        model = BaselineLogisticRegression(
            input_dim=image_size * image_size,
            num_classes=num_classes
        )
    elif model_type == "cnn":
        model = CharacterCNN(
            num_classes=num_classes
        )
    else:
        raise ValueError("model_type must be 'baseline' or 'cnn'")

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


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


def compute_accuracy(preds, labels):
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / len(labels)


def plot_confusion_matrix(labels, preds, model_type, save_path=None):
    os.makedirs("outputs", exist_ok=True)

    if save_path is None:
        save_path = f"outputs/{model_type}_confusion_matrix.png"

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_type.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


def save_misclassified_examples(misclassified, model_type, save_path=None, max_examples=9):
    if not misclassified:
        print("No misclassified examples found.")
        return

    os.makedirs("outputs", exist_ok=True)

    if save_path is None:
        save_path = f"outputs/{model_type}_misclassified_examples.png"

    examples = misclassified[:max_examples]

    cols = 3
    rows = (len(examples) + cols - 1) // cols
    plt.figure(figsize=(10, 10))

    for i, example in enumerate(examples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(example["image"].squeeze(0), cmap="gray")
        plt.title(f"T:{example['true']} P:{example['pred']}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved misclassified examples to {save_path}")


def get_misclassified(preds, labels):
    mistakes = [(i, p, l) for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    return mistakes[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["baseline", "cnn"]
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None
    )
    args = parser.parse_args()

    model_type = args.model

    batch_size = 64
    image_size = 28
    num_classes = 10
    data_dir = "data"

    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = f"models/{model_type}_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating model: {model_type}")
    print(f"Loading weights from: {model_path}")

    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    model = load_model(
        model_type=model_type,
        model_path=model_path,
        device=device,
        image_size=image_size,
        num_classes=num_classes
    )

    preds, labels, misclassified = get_predictions(model, test_loader, device)

    accuracy = compute_accuracy(preds, labels)
    print(f"{model_type.upper()} test accuracy: {accuracy:.4f}")

    plot_confusion_matrix(labels, preds, model_type=model_type)
    save_misclassified_examples(misclassified, model_type=model_type)

    print("\nSample misclassifications:")
    for item in misclassified[:10]:
        print(f"Index {item['index']}: true={item['true']}, pred={item['pred']}")


if __name__ == "__main__":
    main()