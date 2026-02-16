"""
MobileNetV2 student model for deepfake detection.
~3.5M parameters, ~12 MB — the lightweight deployment model.
"""
import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Student(nn.Module):
    """
    MobileNetV2-based deepfake detector (student model).

    Uses torchvision's MobileNetV2 pretrained on ImageNet,
    with a custom binary classification head.
    Designed for mobile/real-time deployment (~3.5M params, ~12 MB).
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()

        # Load MobileNetV2 backbone
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Get feature dimension from the last conv layer
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280

        # Replace the classifier
        self.backbone.classifier = nn.Identity()

        # Custom binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 256, 256)

        Returns:
            logits: Tensor of shape (B,) — raw logits for fake probability
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def get_features(self, x):
        """Extract backbone features (1280-dim)."""
        return self.backbone(x)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_size_mb(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def build_student(pretrained=True, device="cuda"):
    """Factory function to create and configure the student model."""
    model = MobileNetV2Student(pretrained=pretrained)
    total, trainable = model.count_parameters()
    size_mb = model.get_model_size_mb()
    print(f"[Student] MobileNetV2 loaded:")
    print(f"  Total params:     {total / 1e6:.1f}M")
    print(f"  Trainable params: {trainable / 1e6:.1f}M")
    print(f"  Model size:       {size_mb:.1f} MB")
    return model.to(device)
