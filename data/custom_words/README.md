# Custom Word Test Images

Put simple handwritten word images in this folder when manually testing
`src/prediction/predict.py`.

Good first test cases:

- `14AB5.jpeg`
- `happy.jpeg`
- `hello.jpeg`
- `cat.jpeg`

Use clear dark writing on a light background, with letters separated enough for
the segmentation code to detect one box per character.

From the project root, test segmentation first:

```bash
venv/bin/python -m src.segmentation.segment data/custom_words/hello.jpeg
```

Then test prediction:

```bash
venv/bin/python -m src.prediction.predict data/custom_words/hello.jpeg \
  --model-path models/cnn_emnist.pth \
  --dataset emnist
```

The prediction command requires a trained CNN checkpoint at the path provided
with `--model-path`.
