"""
Dataset 1: Chest X-ray Pneumonia Dataset
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Task: Binary Image Classification (0: NORMAL, 1: PNEUMONIA)
Purpose: Baseline model experimentation and binary classification.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

PNEUMONIA_CLASSES = ["NORMAL", "PNEUMONIA"]

class ChestXrayPneumoniaDataset(Dataset):
    """
    Dataset loader for Kaggle Chest X-ray Pneumonia dataset.
    Maintains provenance and avoids patient-level leakage.
    Structure expected:
        root_dir/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    """
    PROVENANCE = {
        "dataset_name": "Chest X-ray Pneumonia Dataset",
        "source": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "task_type": "binary_classification",
        "classes": PNEUMONIA_CLASSES,
        "input_channels": 3,
        "patient_level_split": True,
        "citation": "Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), Large Dataset of Labeled Medical Images."
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = target_size
        self.samples: List[Tuple[Path, int]] = []

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms(split, target_size)

        self._load_samples()

    def default_transforms(self, split: str, size: Tuple[int, int]) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_samples(self):
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            # If no split subfolders, check if root itself contains NORMAL and PNEUMONIA
            split_dir = self.root_dir

        for label_idx, class_name in enumerate(PNEUMONIA_CLASSES):
            class_folder = split_dir / class_name
            if not class_folder.exists():
                # case-insensitive search
                for p in split_dir.glob("*"):
                    if p.is_dir() and p.name.upper() == class_name.upper():
                        class_folder = p
                        break

            if class_folder.exists():
                for ext in ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]:
                    for img_path in class_folder.glob(ext):
                        self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Binary target: tensor of shape [1] with float 0.0 or 1.0 for BCEWithLogitsLoss
        return image, torch.tensor([float(label)], dtype=torch.float32)

    def get_class_counts(self) -> dict:
        counts = {c: 0 for c in PNEUMONIA_CLASSES}
        for _, label in self.samples:
            counts[PNEUMONIA_CLASSES[label]] += 1
        return counts
