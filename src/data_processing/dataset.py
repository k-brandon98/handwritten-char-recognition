import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from src.data_processing.preprocess import get_train_transforms, get_eval_transforms


def load_dataset(dataset_name="mnist", data_dir="data", image_size=28):
    dataset_name = dataset_name.lower()
    train_transform = get_train_transforms(
        image_size=image_size,
        dataset_name=dataset_name
    )
    eval_transform = get_eval_transforms(
        image_size=image_size,
        dataset_name=dataset_name
    )

    if dataset_name == "mnist":
        full_train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=eval_transform
        )

        num_classes = 10

    elif dataset_name == "emnist":
        full_train_dataset = datasets.EMNIST(
            root=data_dir,
            split="balanced",
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.EMNIST(
            root=data_dir,
            split="balanced",
            train=False,
            download=True,
            transform=eval_transform
        )

        num_classes = 47

    elif dataset_name == "emnist_letters":
        target_transform = lambda label: label - 1
        full_train_dataset = datasets.EMNIST(
            root=data_dir,
            split="letters",
            train=True,
            download=True,
            transform=train_transform,
            target_transform=target_transform
        )

        test_dataset = datasets.EMNIST(
            root=data_dir,
            split="letters",
            train=False,
            download=True,
            transform=eval_transform,
            target_transform=target_transform
        )

        num_classes = 26

    elif dataset_name == "emnist_byclass":
        full_train_dataset = datasets.EMNIST(
            root=data_dir,
            split="byclass",
            train=True,
            download=True,
            transform=train_transform
        )

        test_dataset = datasets.EMNIST(
            root=data_dir,
            split="byclass",
            train=False,
            download=True,
            transform=eval_transform
        )

        num_classes = 62

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return full_train_dataset, test_dataset, num_classes


def create_splits(full_train_dataset, val_ratio=0.1, seed=42):
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_subset, val_subset


def get_dataloaders(dataset_name="mnist", data_dir="data", batch_size=64, image_size=28):
    full_train_dataset, test_dataset, num_classes = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size
    )

    train_subset, val_subset = create_splits(full_train_dataset)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes


def print_dataset_info(dataset_name="mnist"):
    full_train_dataset, test_dataset, num_classes = load_dataset(dataset_name=dataset_name)
    sample_image, sample_label = full_train_dataset[0]

    print("Dataset:", dataset_name.upper())
    print("Train size:", len(full_train_dataset))
    print("Test size:", len(test_dataset))
    print("Image shape:", sample_image.shape)
    print("Sample label:", sample_label)
    print("Number of classes:", num_classes)
