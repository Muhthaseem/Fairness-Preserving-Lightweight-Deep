"""
XceptionNet teacher model for deepfake detection.
~71M parameters, ~91 MB — the large high-accuracy model.
"""
import torch
import torch.nn as nn
import timm


class XceptionNet(nn.Module):
    """
    XceptionNet-based deepfake detector (teacher model).

    Uses the timm library's Xception pretrained on ImageNet,
    with a custom binary classification head.
    """

    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()

        # Load Xception backbone from timm
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # Typically 2048

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 1),  # Single output: logit for "fake" probability
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
            logits: Tensor of shape (B, 1) — raw logits for fake probability
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)  # (B,)

    def get_features(self, x):
        """Extract intermediate features (for Grad-CAM or distillation)."""
        return self.backbone(x)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_size_mb(self):
        """Estimate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def build_teacher(pretrained=True, device="cuda"):
    """Factory function to create and configure the teacher model."""
    model = XceptionNet(pretrained=pretrained)
    total, trainable = model.count_parameters()
    size_mb = model.get_model_size_mb()
    print(f"[Teacher] XceptionNet loaded:")
    print(f"  Total params:     {total / 1e6:.1f}M")
    print(f"  Trainable params: {trainable / 1e6:.1f}M")
    print(f"  Model size:       {size_mb:.1f} MB")
    return model.to(device)
