import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model_baseline import BaselineLogisticRegression
from model_cnn import CharacterCNN

from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():

    # -----------------------------
    # Command-line argument
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "cnn"]
    )
    args = parser.parse_args()

    model_type = args.model

    # -----------------------------
    # Config
    # -----------------------------
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 5
    image_size = 28
    num_classes = 10
    data_dir = "data"

    model_save_path = f"models/{model_type}_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training model: {model_type}")

    os.makedirs("models", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    # -----------------------------
    # Choose model
    # -----------------------------
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
        raise ValueError("Invalid model type")

    model = model.to(device)

    # -----------------------------
    # Loss + optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = 0.0

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    # -----------------------------
    # Test evaluation
    # -----------------------------
    model.load_state_dict(
        torch.load(model_save_path, map_location=device)
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_acc:.4f}"
    )


if __name__ == "__main__":
    main()