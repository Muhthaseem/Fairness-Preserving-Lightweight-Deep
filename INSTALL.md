# Installation Guide

## Prerequisites
- **Python 3.12+** (3.12 recommended for best compatibility)
- **NVIDIA GPU** with CUDA support (tested on RTX 4050 6GB)
- **NVIDIA Driver** 591.74+

## Setup

### 1. Create Virtual Environment
```powershell
cd "d:\M3 Projects\DeepFake_Research"
python -m venv venv
.\venv\Scripts\activate
```

> If Python 3.14 causes issues, use Python 3.12:
> ```powershell
> py -3.12 -m venv venv
> .\venv\Scripts\activate
> ```

### 2. Install PyTorch with CUDA
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then install torchvision separately:
```powershell
pip install torchvision
```

> If torchvision fails from PyPI, try nightly:
> ```powershell
> pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
> ```

### 3. Install Remaining Dependencies
```powershell
pip install timm facenet-pytorch opencv-python Pillow pandas numpy scikit-learn matplotlib seaborn streamlit pytorch-grad-cam tqdm
```

### 4. Verify Installation
```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import timm, cv2, pandas, sklearn, matplotlib, tqdm; print('All packages OK')"
```

Expected output:
```
PyTorch: 2.10.0+cu126
CUDA: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
All packages OK
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `torchvision` not found for Python 3.14 | Install torch first, then `pip install torchvision` from PyPI |
| CUDA not detected | Ensure NVIDIA driver ≥ 525, run `nvidia-smi` to verify |
| Out of memory during training | Reduce `batch_size` in `src/config.py` (try 16 or 8) |
| `facenet-pytorch` install fails | `pip install facenet-pytorch --no-deps` then install deps manually |
