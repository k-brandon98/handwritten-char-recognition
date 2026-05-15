import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
from src.prediction.predict import predict_word

app = Flask(
    __name__,
    static_folder="../frontend",
    static_url_path=""
)


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


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
            )
        finally:
            os.unlink(temp_file.name)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
