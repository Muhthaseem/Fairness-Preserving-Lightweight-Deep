from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import sys
import time
import io
import json
import base64
import numpy as np
from PIL import Image
from torchvision import transforms
from gradcam_utils import GradCAM

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
cam_visualizer = None
thresholds = {}

def load_resources():
    global model, thresholds, detector, annotator, cam_visualizer
    # Load Main Deepfake Model
    model = build_student(pretrained=True, device=DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded fair-distilled model from {MODEL_PATH}")
    model.eval()

    # Initialize Grad-CAM Visualizer
    # For MobileNetV2, features[18] is the final 1280-channel conv layer
    cam_visualizer = GradCAM(model, model.backbone.features[18])
    print("Grad-CAM Visualizer initialized")

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

import cv2
import tempfile

# Initialize resources on startup
load_resources()

# --- Image/Video Processing ---
def preprocess_frame(frame_bgr):
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(frame_rgb)
    
    # 1. Detect and Crop Face
    face = detector.mtcnn(face_pil)
    if face is None:
        return None, None, None
        
    # Ensure it's a PIL Image for the annotator
    if isinstance(face, torch.Tensor):
        face_np = face.permute(1, 2, 0).cpu().numpy().astype('uint8')
        face_pil = Image.fromarray(face_np)
    else:
        face_pil = face
        
    # 2. Analyze Demographics
    demo = annotator.predict(face_pil)
    
    # 3. Prepare for Student Model
    input_tensor = annotator.transform(face_pil).unsqueeze(0).to(DEVICE)
    
    # Convert PIL crop back to BGR for Grad-CAM
    face_bgr = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
    
    return input_tensor, demo, face_bgr

def preprocess_image(image_bytes):
    # Load full image
    full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Detect and Crop Face
    face = detector.mtcnn(full_image)
    if face is None:
        return None, None, None 

    # Ensure it's a PIL Image
    if isinstance(face, torch.Tensor):
        face_np = face.permute(1, 2, 0).cpu().numpy().astype('uint8')
        face_pil = Image.fromarray(face_np)
    else:
        face_pil = face
        
    # 2. Analyze Demographics
    demo = annotator.predict(face_pil)
    
    # 3. Prepare for Student Model
    input_tensor = annotator.transform(face_pil).unsqueeze(0).to(DEVICE)
    
    # Convert PIL crop back to BGR for Grad-CAM
    face_bgr = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
    
    return input_tensor, demo, face_bgr

def get_cam_base64(input_tensor, frame_bgr):
    heatmap_bgr = cam_visualizer.generate_heatmap(input_tensor, frame_bgr)
    _, buffer = cv2.imencode('.jpg', heatmap_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample 1 frame per second
    sample_rate = max(1, int(fps))
    frames_processed = 0
    
    # Metadata for the consensus
    frame_results = [] # List of (is_fake, probability, demographic_info, heatmap_candidate)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            input_tensor, demo, face_bgr_crop = preprocess_frame(frame)
            if input_tensor is not None:
                frames_processed += 1
                with torch.no_grad():
                    logit = model(input_tensor)
                    prob = torch.sigmoid(logit).item()
                
                group_name = demo["demographic_group"]
                threshold = thresholds.get(group_name, 0.5)
                
                is_fake = prob > threshold
                
                # We store a potential heatmap for the 'most fake' frame found so far
                heatmap_b64 = None
                if is_fake:
                    heatmap_b64 = get_cam_base64(input_tensor, face_bgr_crop)

                frame_results.append({
                    "is_fake": is_fake,
                    "probability": prob,
                    "demo": {
                        "group": group_name,
                        "race": demo["race"],
                        "gender": demo["gender"],
                        "threshold_applied": round(threshold, 4)
                    },
                    "heatmap": heatmap_b64
                })

        frame_idx += 1
        # Hard limit for demo: 30 seconds of video
        if frame_idx > fps * 30:
            break
            
    cap.release()

    if not frame_results:
        return {"error": "No faces detected in video"}

    # --- Forensic Voting Consensus ---
    fake_count = sum(1 for r in frame_results if r["is_fake"])
    fake_ratio = fake_count / len(frame_results)
    
    # Threshold: If > 30% of frames are fake, the video is FAKE
    # Why 30%? Real videos might have noise/glitches in 5-10% of frames, but deepfakes are usually sustained.
    is_fake_consensus = fake_ratio > 0.30
    
    # Find the 'most significant' frame for the report
    if is_fake_consensus:
        # Most confident Fake frame
        significant_frame = max([r for r in frame_results if r["is_fake"]], key=lambda x: x["probability"])
        prediction = "FAKE"
        # Average probability of the fake frames
        confidence = sum(r["probability"] for r in frame_results if r["is_fake"]) / fake_count
    else:
        # Most confident Real frame
        significant_frame = min(frame_results, key=lambda x: x["probability"])
        prediction = "REAL"
        # Average probability of real frames (converted to confidence)
        real_frames = [r for r in frame_results if not r["is_fake"]]
        if not real_frames: # Edge case: all frames were fake but ratio <= 30%
            confidence = 1 - (sum(r["probability"] for r in frame_results) / len(frame_results))
        else:
            confidence = 1 - (sum(r["probability"] for r in real_frames) / len(real_frames))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probability": significant_frame["probability"],
        "frames_scanned": frames_processed,
        "fake_ratio": round(fake_ratio, 4),
        "demographics": significant_frame["demo"],
        "heatmap": significant_frame["heatmap"],
        "is_video": True
    }

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No media uploaded"}), 400
    
    file = request.files['image']
    filename = file.filename.lower()
    
    start_time = time.time()
    
    # Handle Video
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            
        try:
            result = analyze_video(tmp_path)
            inference_time = (time.time() - start_time) * 1000
            result["inference_ms"] = round(inference_time, 2)
            result["model_info"] = {
                "architecture": "MobileNetV2",
                "weights": "Fair-Distilled-v2",
                "bias_mitigation": "Active (Temporal Scan + Grad-CAM)"
            }
            return jsonify(result)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    # Handle Image
    image_bytes = file.read()
    try:
        input_tensor, demo, face_bgr_crop = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Invalid media: {str(e)}"}), 400

    if input_tensor is None:
        return jsonify({"error": "No face detected. Ensure the media contains clear faces."}), 400

    with torch.no_grad():
        logit = model(input_tensor)
        inference_time = (time.time() - start_time) * 1000
    
    prob = torch.sigmoid(logit).item()
    group_name = demo["demographic_group"]
    optimized_threshold = thresholds.get(group_name, 0.5)
    
    prediction = "FAKE" if prob > optimized_threshold else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    # Generate Heatmap for Image if FAKE (applied to CROP)
    heatmap_b64 = None
    if prediction == "FAKE":
        heatmap_b64 = get_cam_base64(input_tensor, face_bgr_crop)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "probability": prob,
        "inference_ms": round(inference_time, 2),
        "heatmap": heatmap_b64,
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
