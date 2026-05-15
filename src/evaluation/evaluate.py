import os
import string
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import confusion_matrix

from src.data_processing.dataset import get_dataloaders
from src.models.models_cnn import SimpleCNN

def get_label_map(dataset_name):
    if dataset_name == "emnist_letters":
        return {i: chr(ord('A') + i) for i in range(26)}

    elif dataset_name == "mnist":
        return {i: str(i) for i in range(10)}

    elif dataset_name == "emnist_balanced":
        chars = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        return {i: chars[i] for i in range(len(chars))}

    else:
        return None

# Load the model
def load_model(model_path, device, num_classes):
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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


# main evaluation function
def main():
    dataset_name = "emnist_letters"
    batch_size = 64
    image_size = 28
    data_dir = "data"
    model_path = f"models/cnn_{dataset_name}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    print(f"Evaluating CNN on {dataset_name.upper()} with {num_classes} classes")

    model = load_model(model_path, device, num_classes=num_classes)

    preds, labels, misclassified = get_predictions(model, test_loader, device)

    accuracy = compute_accuracy(preds, labels)
    print(f"CNN test accuracy on {dataset_name.upper()}: {accuracy:.4f}")

    plot_confusion_matrix(labels, preds, dataset_name=dataset_name)
    save_misclassified_examples(misclassified, dataset_name=dataset_name)

    print("\nSample misclassifications:")
    for item in misclassified[:10]:
        print(f"Index {item['index']}: true={item['true']}, pred={item['pred']}")


if __name__ == "__main__":
    main()

