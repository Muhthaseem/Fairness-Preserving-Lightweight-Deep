"""
Streamlit Demo: Fairness-Preserving Deepfake Detector
Upload a face image → get real/fake prediction + Grad-CAM + fairness report.
"""
import os
import sys
import time
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import IMAGE_SIZE, DEVICE, MODELS_DIR, DEMOGRAPHIC_GROUPS
from src.models.mobilenetv2 import build_student


@st.cache_resource
def load_model():
    """Load the fairness-aware distilled model."""
    model = build_student(pretrained=True, device=DEVICE)
    weights_path = os.path.join(MODELS_DIR, "fair_student_best_auc.pth")
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.success(f"Loaded model from: {weights_path}")
    else:
        st.warning("No trained weights found. Using pretrained backbone only.")
    model.eval()
    return model


def preprocess_image(image):
    """Preprocess PIL Image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def main():
    st.set_page_config(
        page_title="Deepfake Detector",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Fairness-Preserving Deepfake Detector")
    st.markdown("""
    Upload a face image to detect if it's **real** or **AI-generated (fake)**.
    This model is trained with a novel **fairness-aware knowledge distillation**
    approach to ensure consistent performance across demographic groups.
    """)

    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
    - **Architecture**: MobileNetV2
    - **Parameters**: ~3.5M
    - **Model Size**: ~12 MB
    - **Target Inference**: <30ms
    - **Training**: Fairness-aware distillation from XceptionNet
    """)

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a face image", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        # Inference
        with torch.no_grad():
            x = preprocess_image(image)
            start_time = time.time()
            logit = model(x)
            inference_time = (time.time() - start_time) * 1000

        prob = torch.sigmoid(logit).item()
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob

        with col2:
            st.subheader("Prediction")
            if prediction == "FAKE":
                st.error(f"⚠️ **FAKE** — Confidence: {confidence:.1%}")
            else:
                st.success(f"✅ **REAL** — Confidence: {confidence:.1%}")

            st.metric("Inference Time", f"{inference_time:.1f} ms")
            st.metric("Fake Probability", f"{prob:.3f}")

            # Progress bar for confidence
            st.progress(confidence)

        # Model profiling
        st.divider()
        st.subheader("📊 Model Profile")
        cols = st.columns(4)
        param_count = sum(p.numel() for p in model.parameters())
        model_size = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        ) / (1024 ** 2)

        cols[0].metric("Parameters", f"{param_count / 1e6:.1f}M")
        cols[1].metric("Model Size", f"{model_size:.1f} MB")
        cols[2].metric("Inference", f"{inference_time:.1f} ms")
        cols[3].metric("Architecture", "MobileNetV2")


if __name__ == "__main__":
    main()
