# SecureEye Deepfake Detection - Web App Startup Guide

This guide provides step-by-step instructions to launch the professional forensic web application for the deepfake detection system.

## 🏗️ System Overview
- **Backend**: Flask API (handles MTCNN face detection, FairFace demographic analysis, and Fair-Distilled model inference).
- **Frontend**: Vite + React (Premium glassmorphic dashboard with real-time forensic animations).

---

## 🚀 Quick Start Instructions

Follow these steps in order to ensure the system initializes correctly.

### 1. Start the AI Backend
The backend must be running first so the frontend can connect to the neural engine.

1. Open a terminal in the project root.
2. Activate your virtual environment:
   ```powershell
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```
3. Launch the Flask server:
   ```powershell
   python web/backend/app.py
   ```
   *Verification: You should see "Loaded fair-distilled model" and "Serving Flask app" in the logs.*

### 2. Start the Frontend Dashboard
1. Open a **second** terminal window.
2. Navigate to the frontend directory:
   ```powershell
   cd web/app
   ```
3. Start the development server:
   ```powershell
   npm run dev
   ```
4. Click the URL provided (typically [http://localhost:5175/](http://localhost:5175/)).

---

## 🕵️ How to Perform an Analysis
1. **Ingest Asset**: Drag and drop a face image onto the **Neural Ingestion Zone** (the grid area).
2. **Initiate Scan**: Click the "Initiate Forensic Scan" button.
3. **Review Report**:
   - **Machine Consensus**: The final REAL/FAKE decision.
   - **Detected Group**: The demographic classification used for fairness checking.
   - **Applied Threshold**: The specific scientific threshold ($t_{group}$) applied to this specific person.
   - **Bias Mitigation**: Confirmation that fairness parity was enforced during this analysis.

---

## 🛠️ Troubleshooting
- **Blank Screen**: Ensure you are using the correct port (check the `npm run dev` output). Perform a hard refresh (`Ctrl + Shift + R`).
- **Backend Error**: If you see "Invalid image" errors, ensure the backend has been restarted after the most recent updates.
- **No Face Detected**: MTCNN requires a clear view of the face. Try an image with better lighting or a more direct angle.

---
*Created for the Fairness-Preserving Deepfake Detection Research Project (2026).*
