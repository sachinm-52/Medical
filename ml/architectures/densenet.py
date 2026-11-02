"""
DenseNet121 Architecture for Chest X-ray Classification.
Primary architecture recommended for chest radiography due to feature reuse
and gradient flow through dense connections.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121Model(nn.Module):
    """
    DenseNet121 with customizable classification head for binary or multi-label.
    Supports ImageNet pretrained weights (with disclosure) or training from scratch.
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

        # Load backbone
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = densenet121(weights=weights)

        self.features = base_model.features
        in_features = base_model.classifier.in_features

        # Output dimensions: binary classification uses 1 logit, multi-label uses num_classes logits
        out_features = 1 if task_type == "binary" else num_classes

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = nn.functional.relu(features, inplace=False)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def get_gradcam_target_layer(self) -> nn.Module:
        """Target layer for Grad-CAM gradient backpropagation: the final batchnorm/dense block"""
        return self.features.norm5
