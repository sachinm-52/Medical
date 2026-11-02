"""
Dataset 3: Stanford CheXpert Dataset
Source: https://stanfordmlgroup.github.io/competitions/chexpert/

Task: Multi-label Chest X-ray Classification with Uncertainty Handling
Purpose: Handle uncertain labels appropriately and evaluate model generalization.
"""

import os
import csv
import re
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CHEXPERT_14_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# Standard 5 competition benchmark diseases
CHEXPERT_COMPETITION_5 = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion"
]

class CheXpertDataset(Dataset):
    """
    Dataset loader for Stanford CheXpert.
    Supports uncertain label handling policies:
    - 'u-zeros': treat -1 as 0
    - 'u-ones': treat -1 as 1
    - 'u-ignore': target vector with -1 kept for masked BCE loss
    Avoids patient-level data leakage by grouping by patient ID.
    """
    PROVENANCE = {
        "dataset_name": "Stanford CheXpert",
        "source": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "task_type": "multi_label_classification",
        "classes": CHEXPERT_14_CLASSES,
        "competition_classes": CHEXPERT_COMPETITION_5,
        "input_channels": 3,
        "patient_level_split": True,
        "citation": "Irvin J, Rajpurkar P, Ko M, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI 2019."
    }

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        split: str = "train",
        uncertainty_policy: str = "u-zeros",  # "u-zeros", "u-ones", "u-ignore"
        use_competition_5_only: bool = False,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        random_seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.csv_file = Path(csv_file)
        self.split = split
        self.uncertainty_policy = uncertainty_policy.lower()
        self.use_competition_5_only = use_competition_5_only
        self.target_size = target_size
        self.random_seed = random_seed
        self.classes = CHEXPERT_COMPETITION_5 if use_competition_5_only else CHEXPERT_14_CLASSES

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms(split, target_size)

        self.samples: List[Tuple[Path, List[float], str]] = []  # (img_path, labels, patient_id)
        if self.csv_file.exists():
            self._load_and_split()

    def default_transforms(self, split: str, size: Tuple[int, int]) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor],
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _extract_patient_id(self, path_str: str) -> str:
        # e.g., 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg'
        match = re.search(r"patient\d+", path_str)
        if match:
            return match.group(0)
        return path_str.split("/")[2] if "/" in path_str else "patient_unknown"

    def _process_label(self, val_str: str) -> float:
        if val_str is None or val_str.strip() == "":
            return 0.0
        try:
            val = float(val_str)
        except ValueError:
            return 0.0

        if val == -1.0:
            if self.uncertainty_policy == "u-ones":
                return 1.0
            elif self.uncertainty_policy == "u-zeros":
                return 0.0
            else:  # u-ignore
                return -1.0
        elif val == 1.0:
            return 1.0
        else:
            return 0.0

    def _load_and_split(self):
        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("Path", "")
                patient_id = self._extract_patient_id(rel_path)
                
                # Check path relative to root_dir or absolute
                img_path = self.root_dir / rel_path
                if not img_path.exists():
                    # Check if relative part without dataset prefix exists
                    trimmed = "/".join(rel_path.split("/")[1:])
                    img_path = self.root_dir / trimmed

                label_vector = []
                for cls_name in self.classes:
                    raw_val = row.get(cls_name, "")
                    label_vector.append(self._process_label(raw_val))

                self.samples.append((img_path, label_vector, patient_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_vector, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_vector, dtype=torch.float32)

    def get_provenance_info(self) -> dict:
        info = dict(self.PROVENANCE)
        info["uncertainty_policy"] = self.uncertainty_policy
        info["selected_classes"] = self.classes
        return info
