"""
EfficientNet-B0 Architecture for Chest X-ray Classification.
Compound scaling backbone for efficient, resource-constrained inference.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Model(nn.Module):
    """
    EfficientNet-B0 model with customized classifier.
    Target layer for Grad-CAM is the final feature extraction block: features[-1].
    """
    def __init__(
        self,
        num_classes: int = 14,
        task_type: str = "multi_label",
        pretrained: bool = False,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.task_type = task_type
        self.pretrained = pretrained

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        in_features = base_model.classifier[1].in_features
        out_features = 1 if task_type == "binary" else num_classes

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.features[-1]
