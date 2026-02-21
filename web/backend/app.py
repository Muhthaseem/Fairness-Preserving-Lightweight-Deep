from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import sys
import time
import io
import json
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import IMAGE_SIZE, DEVICE, MODELS_DIR, RESULTS_DIR
from src.models.mobilenetv2 import build_student
from src.data.face_detector import FaceDetector
from src.data.demographic_annotator import FairFaceAnnotator

app = Flask(__name__)
CORS(app)

# --- Configuration & Model Loading ---
MODEL_PATH = os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "v2_test_optimized_thresholds.json")

model = None
detector = None
annotator = None
thresholds = {}

def load_resources():
    global model, thresholds, detector, annotator
    # Load Main Deepfake Model
    model = build_student(pretrained=True, device=DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded fair-distilled model from {MODEL_PATH}")
    model.eval()

    # Load Face Detector (MTCNN)
    detector = FaceDetector(device=DEVICE)
    print("MTCNN Face Detector initialized")
    
    # Load Demographic Annotator (FairFace)
    annotator = FairFaceAnnotator(device=DEVICE)
    print("FairFace Demographic Annotator initialized")

    # Load Optimized Thresholds
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            data = json.load(f)
            # The file has a "thresholds" dict mapping group name to float
            thresholds = data.get("thresholds", data)
            print(f"Loaded {len(thresholds)} optimized thresholds from {THRESHOLD_PATH}")
    else:
        print("No optimized thresholds found, using 0.5 default baseline")

# Initialize resources on startup
load_resources()

# --- Image Processing ---
def preprocess_image(image_bytes):
    # Load full image
    full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Detect and Crop Face (Crucial for correct analysis)
    # detector.mtcnn returns a PIL Image or Tensor [0, 255]
    face = detector.mtcnn(full_image)
    if face is None:
        return None, None 

    # Ensure it's a PIL Image for the annotator
    if isinstance(face, torch.Tensor):
        # MTCNN returns (3, H, W) in [0, 255]
        face_np = face.permute(1, 2, 0).cpu().numpy().astype('uint8')
        face_pil = Image.fromarray(face_np)
    else:
        face_pil = face
        
    # 2. Analyze Demographics (FairFace)
    demo = annotator.predict(face_pil)
    
    # 3. Prepare for Student Model
    # Both use 224x224 and ImageNet normalization
    input_tensor = annotator.transform(face_pil).unsqueeze(0).to(DEVICE)
    
    return input_tensor, demo

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    # Pre-process
    try:
        input_tensor, demo = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    if input_tensor is None:
        return jsonify({"error": "No face detected in the image. Please upload a clear face photo."}), 400

    # Inference (Deepfake Detection)
    start_time = time.time()
    with torch.no_grad():
        logit = model(input_tensor)
        inference_time = (time.time() - start_time) * 1000
    
    prob = torch.sigmoid(logit).item()
    
    # --- Fairness-Aware Analysis ---
    # Apply the optimized threshold for the detected demographic group
    group_name = demo["demographic_group"]
    optimized_threshold = thresholds.get(group_name, 0.5)
    
    prediction = "FAKE" if prob > optimized_threshold else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob # Standard confidence for UI

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "probability": prob,
        "inference_ms": round(inference_time, 2),
        "demographics": {
            "group": group_name,
            "race": demo["race"],
            "gender": demo["gender"],
            "threshold_applied": round(optimized_threshold, 4)
        },
        "model_info": {
            "architecture": "MobileNetV2",
            "weights": "Fair-Distilled-v2",
            "bias_mitigation": "Active (Threshold Calibration)"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
