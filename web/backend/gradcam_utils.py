import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Grad-CAM implementation for MobileNetV2.
    Highlights regions in the image that contribute most to the 'FAKE' prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, original_image_np):
        """
        Generates a Grad-CAM heatmap overlaid on the original image.
        
        Args:
            input_tensor: Preprocessed tensor (1, 3, 224, 224)
            original_image_np: Original BGR image from OpenCV
            
        Returns:
            Overlaid heatmap image as a BGR numpy array
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for the logit (we want to see what makes it FAKE)
        output.backward()
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # We only care about positive influences on 'FAKE'
        
        # Normalize
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Resize to original image size
        cam_resized = cv2.resize(cam, (original_image_np.shape[1], original_image_np.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlaid_image = cv2.addWeighted(original_image_np, 0.6, heatmap, 0.4, 0)
        
        return overlaid_image
