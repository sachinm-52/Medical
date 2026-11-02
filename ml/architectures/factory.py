"""
Model Factory for MedVision AI.
Provides unified model instantiation, checkpoint loading, and target layer retrieval.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from ml.architectures.densenet import DenseNet121Model
from ml.architectures.resnet import ResNet50Model
from ml.architectures.efficientnet import EfficientNetB0Model

ARCHITECTURE_REGISTRY = {
    "DenseNet121": DenseNet121Model,
    "ResNet50": ResNet50Model,
    "EfficientNet-B0": EfficientNetB0Model,
}

def create_model(
    architecture: str = "DenseNet121",
    num_classes: int = 14,
    task_type: str = "multi_label",
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory method to create and optionally load weights for an architecture.
    """
    if architecture not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {list(ARCHITECTURE_REGISTRY.keys())}"
        )

    model_cls = ARCHITECTURE_REGISTRY[architecture]
    model = model_cls(
        num_classes=num_classes,
        task_type=task_type,
        pretrained=pretrained
    )

    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Strip possible 'module.' prefix from DataParallel training
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            clean_state_dict[key] = v

        model.load_state_dict(clean_state_dict, strict=False)

    if device:
        model.to(device)

    model.eval()
    return model
