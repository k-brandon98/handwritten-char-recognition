from torchvision import transforms


def _get_emnist_orientation_transform(dataset_name):
    if not dataset_name.lower().startswith("emnist"):
        return []

    return [
        transforms.Lambda(
            lambda image: transforms.functional.hflip(
                transforms.functional.rotate(image, -90)
            )
        )
    ]


def get_train_transforms(image_size=28, dataset_name="mnist"):
    return transforms.Compose([
        *_get_emnist_orientation_transform(dataset_name),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def get_eval_transforms(image_size=28, dataset_name="mnist"):
    return transforms.Compose([
        *_get_emnist_orientation_transform(dataset_name),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
