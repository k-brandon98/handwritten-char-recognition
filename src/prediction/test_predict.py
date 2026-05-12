"""
Tests for the word prediction pipeline.

These tests focus on the programming logic in ``predict.py``:

- class-label lookup for MNIST and EMNIST
- image crop preprocessing before CNN inference
- combining individual character predictions into a word
- wiring segmentation + model loading + prediction together

The end-to-end test intentionally uses a temporary randomly initialized CNN
checkpoint. It does not test model accuracy. It only verifies that the pipeline
can load a CNN checkpoint, segment a simple image, predict each segment, and
return a combined string.

Run from the project root:

    venv/bin/python -m unittest src.prediction.test_predict
"""

import os
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.models.models_cnn import SimpleCNN
from src.prediction.predict import (
    CharacterPrediction,
    combine_predictions,
    get_class_names,
    predict_word,
    preprocess_character_crop,
)


class PredictPipelineTests(unittest.TestCase):
    def test_get_class_names_returns_expected_dataset_labels(self):
        self.assertEqual(get_class_names("mnist"), [str(i) for i in range(10)])

        emnist_classes = get_class_names("emnist")
        self.assertEqual(len(emnist_classes), 47)
        self.assertEqual(emnist_classes[:10], [str(i) for i in range(10)])
        self.assertIn("A", emnist_classes)
        self.assertIn("t", emnist_classes)

    def test_get_class_names_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError):
            get_class_names("unknown")

    def test_preprocess_character_crop_matches_cnn_input_shape_and_range(self):
        device = torch.device("cpu")
        crop = np.zeros((28, 28), dtype=np.uint8)
        crop[8:20, 10:18] = 255

        tensor = preprocess_character_crop(crop, device)

        self.assertEqual(tuple(tensor.shape), (1, 1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(tensor.min().item(), -1.0)
        self.assertAlmostEqual(tensor.max().item(), 1.0)

    def test_preprocess_character_crop_rejects_color_images(self):
        device = torch.device("cpu")
        color_crop = np.zeros((28, 28, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            preprocess_character_crop(color_crop, device)

    def test_combine_predictions_joins_labels_in_order(self):
        predictions = [
            CharacterPrediction(index=0, class_index=12, label="C", confidence=0.91),
            CharacterPrediction(index=1, class_index=36, label="a", confidence=0.88),
            CharacterPrediction(index=2, class_index=46, label="t", confidence=0.94),
        ]

        self.assertEqual(combine_predictions(predictions), "Cat")

    def test_predict_word_runs_full_pipeline_with_synthetic_inputs(self):
        """
        Smoke-test the full prediction path without needing a trained model.

        The checkpoint is random, so the predicted labels are not meaningful.
        The assertion is only that the system can produce one label per detected
        segment and combine those labels into a string.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "simple_word.png")
            model_path = os.path.join(temp_dir, "random_cnn.pth")

            image = np.full((80, 180, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                "HI",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 0, 0),
                4,
            )
            cv2.imwrite(image_path, image)

            model = SimpleCNN(num_classes=47)
            torch.save(model.state_dict(), model_path)

            word, predictions = predict_word(
                image_path=image_path,
                model_path=model_path,
                dataset_name="emnist",
                device_name="cpu",
            )

            self.assertIsInstance(word, str)
            self.assertGreaterEqual(len(predictions), 1)
            self.assertEqual(len(word), len(predictions))

            for prediction in predictions:
                self.assertIsNotNone(prediction.box)
                self.assertGreaterEqual(prediction.confidence, 0.0)
                self.assertLessEqual(prediction.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
