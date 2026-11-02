"""
Dataset 4: MedMNIST ChestMNIST Subset
Source: https://medmnist.com/
Zenodo: https://zenodo.org/records/6496656/files/chestmnist.npz

Task: Multi-label Chest X-ray Classification (14 classes)
Purpose: Lightweight model experiments, benchmarking, testing and comparing model performance.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import urllib.request

CHESTMNIST_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia"
]

ZENODO_CHESTMNIST_URL = "https://zenodo.org/records/6496656/files/chestmnist.npz?download=1"

class ChestMNISTDataset(Dataset):
    """
    Dataset loader for ChestMNIST (MedMNIST v2).
    Lightweight 28x28 chest X-ray multi-label dataset with 14 classes.
    Natively partitioned into train (78,468), val (11,219), test (22,433).
    Supports automatic download if missing.
    """
    PROVENANCE = {
        "dataset_name": "MedMNIST v2 (ChestMNIST)",
        "source": "https://medmnist.com/",
        "zenodo_url": "https://zenodo.org/record/6496656",
        "task_type": "multi_label_classification",
        "classes": CHESTMNIST_CLASSES,
        "num_classes": 14,
        "input_channels": 1,
        "standard_channels": 3,  # Converted to 3-channel RGB for DenseNet/ResNet/EfficientNet
        "native_resolution": (28, 28),
        "patient_level_split": True,
        "splits": {"train": 78468, "val": 11219, "test": 22433},
        "citation": "Yang J, Shi R, Wei D, et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 2023."
    }

    def __init__(
        self,
        root_dir: str = "./data/chestmnist",
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.split = split  # 'train', 'val', or 'test'
        self.target_size = target_size
        self.npz_path = self.root_dir / "chestmnist.npz"

        if download and not self.npz_path.exists():
            self._download_npz()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms(split, target_size)

        self._load_data()

    def _download_npz(self):
        print(f"[ChestMNIST] Downloading dataset from Zenodo to {self.npz_path}...")
        try:
            # First try medmnist library if available
            import medmnist
            from medmnist import ChestMNIST
            med_ds = ChestMNIST(split="train", download=True, root=str(self.root_dir))
        except Exception as e:
            # Fallback to direct url download
            urllib.request.urlretrieve(ZENODO_CHESTMNIST_URL, str(self.npz_path))
            print(f"[ChestMNIST] Download complete: {self.npz_path}")

    def _load_data(self):
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"ChestMNIST dataset file not found at {self.npz_path}. "
                f"Please enable download=True or place chestmnist.npz in {self.root_dir}."
            )
        data = np.load(str(self.npz_path))
        split_key_img = f"{self.split}_images"
        split_key_lbl = f"{self.split}_labels"

        if split_key_img not in data:
            raise KeyError(f"Key {split_key_img} not in {list(data.keys())}")

        self.images = data[split_key_img]  # shape: (N, 28, 28)
        self.labels = data[split_key_lbl]  # shape: (N, 14)

    def default_transforms(self, split: str, size: Tuple[int, int]) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # MedMNIST 28x28 is grayscale, duplicate across 3 channels for backbone models
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image_tensor = self.transform(img_arr)
        else:
            image_tensor = torch.from_numpy(img_arr).float() / 255.0

        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def get_class_counts(self) -> Dict[str, int]:
        summed = np.sum(self.labels, axis=0)
        return {CHESTMNIST_CLASSES[i]: int(summed[i]) for i in range(len(CHESTMNIST_CLASSES))}
