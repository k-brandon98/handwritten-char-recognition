from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from src.prediction.predict import predict_word
# import your existing prediction function
# example:
# from src.prediction.predict import predict_image

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Handwriting recognition backend is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("L")

    # Replace this with your real prediction function
    # prediction = predict_image(image)
    prediction, _ = predict_word(image)  

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)