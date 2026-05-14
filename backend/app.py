import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from src.prediction.predict import predict_word

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"]}},
)


@app.route("/")
def home():
    return jsonify({"message": "Handwriting recognition backend is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("L")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        try:
            prediction, _ = predict_word(
                temp_file.name,
                model_path="models/cnn_emnist.pth",
                dataset_name="emnist",
            )
        finally:
            os.unlink(temp_file.name)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
