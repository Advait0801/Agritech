from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
import gdown
import os

app = Flask(__name__)
CORS(app)

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/disease_model.pth"
GDRIVE_URL = "https://drive.google.com/uc?id=14-23-b4YtxMpg_rnom65whN3rbz6rsCI"

if not os.path.exists(MODEL_PATH):
    print("Downloading disease_model.pth from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load disease prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disease_model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=5
)
disease_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
disease_model.to(device)
disease_model.eval()

class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    return input_tensor

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    image = Image.open(request.files['file'].stream)
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = disease_model(input_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence_score, predicted_class = torch.max(probabilities, 1)

    return jsonify({
        'predicted_class': class_names[predicted_class.item()],
        'confidence_score': confidence_score.item()
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)