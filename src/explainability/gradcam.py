"""
Grad-CAM visualization for deepfake detection models.
Generates heatmaps showing which facial regions the model focuses on.
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import IMAGE_SIZE, GRADCAM_DIR, DEVICE


class GradCAM:
    """
    Grad-CAM implementation for binary classification models.
    Targets the last convolutional layer to generate class activation maps.
    """

    def __init__(self, model, target_layer=None, device=DEVICE):
        """
        Args:
            model: The CNN model
            target_layer: The convolutional layer to target. If None, auto-detect.
            device: Computation device
        """
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)

        self.target_layer = target_layer

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d,)):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model")
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, class_idx=None):
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: PIL Image or tensor
            class_idx: Target class (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()

        if isinstance(image, Image.Image):
            x = self.transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                x = image.unsqueeze(0).to(self.device)
            else:
                x = image.to(self.device)
        else:
            raise ValueError("Input must be PIL Image or Tensor")

        # Forward pass
        x.requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = (torch.sigmoid(logits) > 0.5).long().item()

        # Backward pass
        self.model.zero_grad()
        target = logits if class_idx == 1 else -logits
        target.backward()

        # Grad-CAM computation
        gradients = self.gradients[0]     # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        return cam

    def generate_overlay(self, image, alpha=0.5):
        """
        Generate a heatmap overlay on the original image.

        Args:
            image: PIL Image
            alpha: Overlay transparency

        Returns:
            overlay: numpy array (H, W, 3) in RGB
            heatmap: numpy array (H, W) normalized
        """
        heatmap = self.generate(image)

        # Convert image to numpy
        img = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)

        return overlay, heatmap


def generate_gradcam_per_group(model, test_csv, num_samples_per_group=5,
                                output_dir=GRADCAM_DIR, device=DEVICE):
    """
    Generate Grad-CAM visualizations for sample images from each demographic group.

    Args:
        model: Trained model
        test_csv: Path to test set CSV with face_path, group_id, demographic_group
        num_samples_per_group: Number of images per group to visualize
        output_dir: Directory to save visualizations
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(test_csv)
    gradcam = GradCAM(model, device=device)

    for group_name in df["demographic_group"].unique():
        group_df = df[df["demographic_group"] == group_name]
        samples = group_df.sample(n=min(num_samples_per_group, len(group_df)),
                                   random_state=42)

        fig, axes = plt.subplots(2, num_samples_per_group,
                                  figsize=(4*num_samples_per_group, 8))
        fig.suptitle(f"Grad-CAM: {group_name}", fontsize=16)

        for i, (_, row) in enumerate(samples.iterrows()):
            if i >= num_samples_per_group:
                break

            try:
                img = Image.open(row["face_path"]).convert("RGB")
            except Exception:
                continue

            overlay, heatmap = gradcam.generate_overlay(img)

            # Original image
            axes[0, i].imshow(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            axes[0, i].set_title(row["label"], fontsize=10)
            axes[0, i].axis("off")

            # Grad-CAM overlay
            axes[1, i].imshow(overlay)
            axes[1, i].set_title("Grad-CAM", fontsize=10)
            axes[1, i].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"gradcam_{group_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nGrad-CAM visualizations saved to: {output_dir}")
