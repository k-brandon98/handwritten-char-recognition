# Context-aware handwritten word prediction using Wordfreq

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from wordfreq import zipf_frequency

from src.models.models_cnn import SimpleCNN
from src.models.model_baseline import BaselineLogisticRegression
from src.segmentation.segment import segment_word

MNIST_CLASSES = [str(i) for i in range(10)]
EMNIST_LETTER_CLASSES = [chr(ord("a") + index) for index in range(26)]
EMNIST_BYCLASS_CLASSES = (
    MNIST_CLASSES
    + [chr(ord("A") + index) for index in range(26)]
    + EMNIST_LETTER_CLASSES
)

EMNIST_BALANCED_CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "n",
    "q",
    "r",
    "t",
]

CONFUSABLES = {
    "0": ["o", "O"],
    "o": ["0"],
    "O": ["0"],
    "1": ["l", "I"],
    "l": ["1", "I"],
    "I": ["1", "l"],
    "5": ["s", "S"],
    "s": ["5"],
    "S": ["5"],
    "2": ["z", "Z"],
    "z": ["2"],
    "Z": ["2"],
    "U": ["u"],
}


@dataclass
class CharacterPrediction:
    index: int
    class_index: int
    label: str
    confidence: float
    alternatives: List[Tuple[str, float]]
    box: Optional[Tuple[int, int, int, int]] = None


def get_class_names(dataset_name: str) -> List[str]:
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return MNIST_CLASSES

    if dataset_name == "emnist":
        return EMNIST_BALANCED_CLASSES

    if dataset_name == "emnist_letters":
        return EMNIST_LETTER_CLASSES

    if dataset_name == "emnist_byclass":
        return EMNIST_BYCLASS_CLASSES

    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def get_device(device_name: Optional[str] = None) -> torch.device:
    if device_name:
        return torch.device(device_name)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(
    model_path: str,
    num_classes: int,
    device: torch.device,
):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    # Detect model architecture from checkpoint keys
    if "linear.weight" in state_dict:
        # Baseline logistic regression model
        model = BaselineLogisticRegression(num_classes=num_classes).to(device)
    else:
        # SimpleCNN model
        model = SimpleCNN(num_classes=num_classes).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_character_crop(
    char_image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:

    tensor = torch.from_numpy(char_image.astype(np.float32)) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(device)


def predict_character_topk(
    model: SimpleCNN,
    char_image: np.ndarray,
    class_names: Sequence[str],
    device: torch.device,
    top_k: int = 5,
):

    input_tensor = preprocess_character_crop(char_image, device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=top_k)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        label = class_names[idx.item()]
        results.append((label, prob.item()))

        if label in CONFUSABLES:
            for alt in CONFUSABLES[label]:
                results.append((alt, prob.item() * 0.85))

    dedup = {}
    for label, score in results:
        dedup[label] = max(score, dedup.get(label, 0))

    sorted_results = sorted(dedup.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:top_k]


def predict_characters(
    model: SimpleCNN,
    character_images: Iterable[np.ndarray],
    class_names: Sequence[str],
    device: torch.device,
    boxes=None,
):

    predictions = []

    for index, char_image in enumerate(character_images):

        alternatives = predict_character_topk(
            model=model,
            char_image=char_image,
            class_names=class_names,
            device=device,
            top_k=5,
        )

        label, confidence = alternatives[0]

        class_index = class_names.index(label) if label in class_names else -1

        box = boxes[index] if boxes is not None else None

        predictions.append(
            CharacterPrediction(
                index=index,
                class_index=class_index,
                label=label,
                confidence=confidence,
                alternatives=alternatives,
                box=box,
            )
        )

    return predictions


def score_candidate(candidate: str, char_predictions):

    cnn_score = 0.0

    for i, char in enumerate(candidate):
        probs = dict(char_predictions[i].alternatives)
        cnn_score += np.log(probs.get(char, 1e-8))

    wordfreq_score = zipf_frequency(candidate.lower(), "en")

    return cnn_score + (0.4 * wordfreq_score)


def combine_predictions_contextual(
    predictions,
    beam_width=3,
):

    candidate_lists = []

    for pred in predictions:
        candidate_lists.append([label for label, _ in pred.alternatives[:beam_width]])

    candidates = itertools.product(*candidate_lists)

    best_word = None
    best_score = float("-inf")

    for chars in candidates:
        word = "".join(chars)

        score = score_candidate(word, predictions)

        if score > best_score:
            best_score = score
            best_word = word

    return best_word


def predict_word(
    image_path: str,
    model_path: str = "models/cnn_emnist_byclass.pth",
    dataset_name: str = "emnist_byclass",
    output_size: int = 28,
    device_name: Optional[str] = None,
):

    class_names = get_class_names(dataset_name)

    device = get_device(device_name)

    model = load_trained_model(
        model_path=model_path,
        num_classes=len(class_names),
        device=device,
    )

    characters, boxes, _, _ = segment_word(image_path, output_size=output_size)

    if not characters:
        raise ValueError("No characters detected.")

    predictions = predict_characters(
        model=model,
        character_images=characters,
        class_names=class_names,
        device=device,
        boxes=boxes,
    )

    word = combine_predictions_contextual(predictions)

    return word, predictions


def build_arg_parser():

    parser = argparse.ArgumentParser(
        description="Context-aware handwritten word prediction."
    )

    parser.add_argument("image_path", help="Path to handwritten word image.")

    parser.add_argument(
        "--model-path",
        default="models/cnn_emnist_byclass.pth",
    )

    parser.add_argument(
        "--dataset",
        default="emnist_byclass",
        choices=[
            "mnist",
            "emnist",
            "emnist_letters",
            "emnist_byclass",
        ],
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    return parser


def main():

    args = build_arg_parser().parse_args()

    word, predictions = predict_word(
        image_path=args.image_path,
        model_path=args.model_path,
        dataset_name=args.dataset,
        device_name=args.device,
    )

    print(f"\nFinal Word Prediction: {word}\n")

    for pred in predictions:
        print(
            f"Character {pred.index}: "
            f"{pred.label} "
            f"(confidence={pred.confidence:.4f})"
        )

        print("Alternatives:")

        for alt, score in pred.alternatives:
            print(f"  {alt}: {score:.4f}")

        print()


if __name__ == "__main__":
    main()
