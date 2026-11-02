"""
ResNet50 Architecture for Chest X-ray Classification.
Residual network backbone for comparative benchmarking against DenseNet121.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Model(nn.Module):
    """
    ResNet50 with custom classification head.
    Target layer for Grad-CAM is layer4[-1].
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

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_model = resnet50(weights=weights)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        in_features = base_model.fc.in_features
        out_features = 1 if task_type == "binary" else num_classes

        self.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.layer4[-1]
