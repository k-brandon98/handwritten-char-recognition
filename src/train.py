import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model_baseline import BaselineLogisticRegression
# from model_baseline import BaselineMLP

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
    # Config
    # -----------------------------
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 5
    val_split = 0.1
    model_save_path = "models/baseline_emnist.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # -----------------------------
    # Minimal preprocessing
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize to roughly centered range
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # -----------------------------
    # Load dataset
    # -----------------------------
    full_train_dataset = datasets.EMNIST(
      root="data",
      split="digits",   # or "letters", "balanced", "byclass"
      train=True,
      download=True,
      transform=transform
    )

    test_dataset = datasets.EMNIST(
        root="data",
        split="digits",
        train=False,
        download=True,
        transform=transform
    )

    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = BaselineLogisticRegression(input_dim=28*28, num_classes=26).to(device)
    # model = BaselineMLP(input_dim=28*28, hidden_dim=128, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    # -----------------------------
    # Final test evaluation
    # -----------------------------
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()