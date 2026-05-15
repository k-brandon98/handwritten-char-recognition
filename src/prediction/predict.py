"""
Predict words from segmented handwritten character images.

This module connects two parts of the project:

1. ``src.segmentation.segment.segment_word`` extracts individual character
   crops from a full word image.
2. ``src.models.models_cnn.SimpleCNN`` classifies each 28x28 crop.

The high-level flow is:

    word image -> segmentation -> character crops -> CNN predictions -> word

Run from the project root with:

    python -m src.prediction.predict data/custom_words/cat.jpeg

By default this expects an EMNIST byclass CNN checkpoint at
``models/cnn_emnist_byclass.pth``. Use ``--dataset`` and ``--model-path`` if
you are using a different checkpoint.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.models_cnn import SimpleCNN
from src.segmentation.segment import segment_word


MNIST_CLASSES = [str(i) for i in range(10)]
EMNIST_LETTER_CLASSES = [chr(ord("a") + index) for index in range(26)]
EMNIST_BYCLASS_CLASSES = (
    MNIST_CLASSES
    + [chr(ord("A") + index) for index in range(26)]
    + EMNIST_LETTER_CLASSES
)

# TorchVision's class order for the EMNIST "balanced" split. The model output
# index must be mapped through this list before joining predictions into text.
EMNIST_BALANCED_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "a", "b", "d", "e",
    "f", "g", "h", "n", "q", "r", "t",
]


@dataclass
class CharacterPrediction:
    """
    Prediction details for one segmented character.

    Attributes:
        index: Character position in the word, after left-to-right sorting.
        class_index: Raw integer class predicted by the CNN.
        label: Human-readable class label, such as "A" or "7".
        confidence: Softmax probability for the predicted class.
        box: Optional segmentation bounding box in ``(x, y, width, height)``
            format. This is useful when debugging segmentation/prediction
            mismatches.
    """

    index: int
    class_index: int
    label: str
    confidence: float
    box: Optional[Tuple[int, int, int, int]] = None


def get_class_names(dataset_name: str) -> List[str]:
    """
    Return the class labels used by a trained model.

    The order of this list must match the order used during training. For this
    project, training currently targets MNIST, EMNIST balanced, EMNIST
    letters, or EMNIST byclass.
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return MNIST_CLASSES

    if dataset_name == "emnist":
        return EMNIST_BALANCED_CLASSES

    if dataset_name == "emnist_letters":
        return EMNIST_LETTER_CLASSES

    if dataset_name == "emnist_byclass":
        return EMNIST_BYCLASS_CLASSES

    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Expected 'mnist', 'emnist', "
        "'emnist_letters', or 'emnist_byclass'."
    )


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Pick the device used for model inference.

    If ``device_name`` is supplied, it is passed directly to ``torch.device``.
    Otherwise CUDA is used when available, falling back to CPU.
    """

    if device_name:
        return torch.device(device_name)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(
    model_path: str,
    num_classes: int,
    device: torch.device,
) -> SimpleCNN:
    """
    Load a trained ``SimpleCNN`` checkpoint for inference.

    Args:
        model_path: Path to a saved PyTorch checkpoint. The training script
            saves a plain ``state_dict``, but this function also accepts a
            checkpoint dictionary with a ``model_state_dict`` key.
        num_classes: Number of output classes expected from the model.
        device: CPU or GPU device where the model should run.

    Returns:
        A ``SimpleCNN`` in evaluation mode.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train the CNN first or pass --model-path."
        )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(
        checkpoint, dict
    ) else checkpoint

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_character_crop(
    char_image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert one segmented character crop into CNN input format.

    ``segment_word`` returns each character as a 28x28 NumPy array with pixel
    values in ``[0, 255]`` and white handwriting on a black background. The CNN
    was trained with TorchVision's ``ToTensor`` and ``Normalize((0.5,), (0.5,))``,
    so this function reproduces that preprocessing:

        uint8 [0, 255] -> float [0, 1] -> normalized [-1, 1]

    The returned tensor has shape ``(1, 1, 28, 28)``:

        batch dimension, channel dimension, height, width
    """

    if char_image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale 2D character crop, got shape {char_image.shape}."
        )

    tensor = torch.from_numpy(char_image.astype(np.float32)) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def predict_character(
    model: SimpleCNN,
    char_image: np.ndarray,
    class_names: Sequence[str],
    device: torch.device,
) -> Tuple[int, str, float]:
    """
    Predict one segmented character image.

    Returns:
        ``(class_index, label, confidence)`` where ``label`` is the mapped class
        name and ``confidence`` is the predicted softmax probability.
    """

    input_tensor = preprocess_character_crop(char_image, device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    class_index = predicted_index.item()

    if class_index >= len(class_names):
        raise IndexError(
            f"Model predicted class index {class_index}, but only "
            f"{len(class_names)} class labels were provided."
        )

    return class_index, class_names[class_index], confidence.item()


def predict_characters(
    model: SimpleCNN,
    character_images: Iterable[np.ndarray],
    class_names: Sequence[str],
    device: torch.device,
    boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> List[CharacterPrediction]:
    """
    Predict every segmented character crop in order.

    Args:
        model: Loaded CNN model.
        character_images: Ordered 28x28 grayscale character crops.
        class_names: Labels corresponding to model output indices.
        device: CPU or GPU device.
        boxes: Optional segmentation boxes aligned with ``character_images``.

    Returns:
        A list of ``CharacterPrediction`` objects.
    """

    predictions = []

    for index, char_image in enumerate(character_images):
        class_index, label, confidence = predict_character(
            model=model,
            char_image=char_image,
            class_names=class_names,
            device=device,
        )
        box = boxes[index] if boxes is not None else None
        predictions.append(
            CharacterPrediction(
                index=index,
                class_index=class_index,
                label=label,
                confidence=confidence,
                box=box,
            )
        )

    return predictions


def combine_predictions(predictions: Sequence[CharacterPrediction]) -> str:
    """
    Combine ordered character predictions into a complete word/string.
    """

    return "".join(prediction.label for prediction in predictions)


def predict_word(
    image_path: str,
    model_path: str = "models/cnn_emnist_byclass.pth",
    dataset_name: str = "emnist_byclass",
    output_size: int = 28,
    device_name: Optional[str] = None,
) -> Tuple[str, List[CharacterPrediction]]:
    """
    Segment a word image, classify each character, and combine the result.

    Args:
        image_path: Path to an image containing a handwritten word.
        model_path: Path to a trained CNN checkpoint.
        dataset_name: Dataset label mapping to use: ``"emnist_byclass"``,
            ``"emnist_letters"``, ``"emnist"``, or ``"mnist"``.
        output_size: Size passed to segmentation. The CNN expects ``28``.
        device_name: Optional device override, such as ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(word, predictions)`` where ``word`` is the combined string and
        ``predictions`` contains per-character details.
    """

    class_names = get_class_names(dataset_name)
    device = get_device(device_name)
    model = load_trained_model(
        model_path=model_path,
        num_classes=len(class_names),
        device=device,
    )

    characters, boxes, _, _ = segment_word(image_path, output_size=output_size)

    if not characters:
        raise ValueError(
            f"No characters were detected in {image_path}. "
            "Check the input image or segmentation thresholds."
        )

    predictions = predict_characters(
        model=model,
        character_images=characters,
        class_names=class_names,
        device=device,
        boxes=boxes,
    )
    return combine_predictions(predictions), predictions


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for word prediction.
    """

    parser = argparse.ArgumentParser(
        description="Predict a handwritten word using segmentation and a CNN."
    )
    parser.add_argument(
        "image_path",
        help="Path to an image containing a handwritten word.",
    )
    parser.add_argument(
        "--model-path",
        default="models/cnn_emnist_byclass.pth",
        help="Path to the trained CNN checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        default="emnist_byclass",
        choices=["mnist", "emnist", "emnist_letters", "emnist_byclass"],
        help="Class mapping used by the trained model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example 'cpu' or 'cuda'.",
    )
    return parser


def main() -> None:
    """
    Run word prediction from the command line.

    This function parses CLI arguments, calls ``predict_word``, and prints both
    the per-character predictions and the final reconstructed word.
    """

    args = build_arg_parser().parse_args()
    word, predictions = predict_word(
        image_path=args.image_path,
        model_path=args.model_path,
        dataset_name=args.dataset,
        device_name=args.device,
    )

    print(f"Detected {len(predictions)} characters")
    for prediction in predictions:
        print(
            f"{prediction.index}: "
            f"label={prediction.label} "
            f"class={prediction.class_index} "
            f"confidence={prediction.confidence:.4f} "
            f"box={prediction.box}"
        )
    print(f"Word: {word}")


if __name__ == "__main__":
    main()
