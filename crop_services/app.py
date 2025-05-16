from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import gdown
import os

app = Flask(__name__)
CORS(app)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

FILE_PATHS = {
    "yield_model": "models/yield_model.pkl",
    "recommendation_model": "models/recommendation_model.pkl"
}

GDRIVE_URLS = {
    "yield_model": "https://drive.google.com/uc?id=1_ZL7np0WAvBc4ygS5XjmbVVgDxDKRsog",
    "recommendation_model": "https://drive.google.com/uc?id=1RwrEpixMovkgfSl66PBoHm34s6yn1WYg"
}

for key, url in GDRIVE_URLS.items():
    if not os.path.exists(FILE_PATHS[key]):
        print(f"Downloading {key} from Google Drive...")
        gdown.download(url, FILE_PATHS[key], quiet=False)

# Load models
with open(FILE_PATHS["yield_model"], "rb") as f:
    yield_model, yield_scaler = pickle.load(f)

with open(FILE_PATHS["recommendation_model"], "rb") as f:
    crop_model = pickle.load(f)

@app.route("/")
def home():
    return "Crop Services Backend is running!"

@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    try:
        data = request.json
        features = np.array([[data["Soil_Temp"], data["N"], data["P"], data["K"],
                              data["Moisture"], data["Humidity"], data["Air_Temp"]]])
        scaled = yield_scaler.transform(features)
        prediction = yield_model.predict(scaled)[0]
        return jsonify({"yield": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.json
        features = [data[key] for key in ['N', 'P', 'K', 'temperature', 'humidity']]
        probabilities = crop_model.predict_proba([features])[0]
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        predictions = [{
            "crop": crop_model.classes_[i],
            "probability": probabilities[i] * 100
        } for i in top3_indices]
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)