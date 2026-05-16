import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
from src.prediction.predict import predict_word as predict_word_no_context
from src.prediction.predict_with_wordfreq import predict_word as predict_word_with_context

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

    # Get optional parameters
    use_context = request.form.get("use_context", "true").lower() == "true"
    dataset_name = request.form.get("dataset", "emnist_byclass")
    model_path = request.form.get("model_path", "models/cnn_emnist_byclass.pth")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        try:
            if use_context:
                prediction, _ = predict_word_with_context(
                    temp_file.name,
                    model_path=model_path,
                    dataset_name=dataset_name,
                )
            else:
                prediction, _ = predict_word_no_context(
                    temp_file.name,
                    model_path=model_path,
                    dataset_name=dataset_name,
                )
        finally:
            os.unlink(temp_file.name)

    return jsonify({
        "prediction": prediction,
        "use_context": use_context,
        "dataset": dataset_name
    })


if __name__ == "__main__":
    app.run(debug=True)
