"""
Dataset 2: NIH ChestX-ray14 Dataset
Source: https://nihcc.app.box.com/v/ChestXray-NIHCC

Task: Multi-label Thoracic Disease Classification (14 diseases)
Purpose: Detect the 14 available disease labels without patient leakage.
"""

import os
import csv
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict, Set
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

NIH_14_CLASSES = [
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

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(NIH_14_CLASSES)}

class NIHChestXray14Dataset(Dataset):
    """
    Dataset loader for NIH ChestX-ray14.
    Implements strict patient-level splitting to avoid data leakage.
    Parses Data_Entry_2017.csv.
    """
    PROVENANCE = {
        "dataset_name": "NIH ChestX-ray14",
        "source": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "task_type": "multi_label_classification",
        "classes": NIH_14_CLASSES,
        "input_channels": 3,
        "patient_level_split": True,
        "citation": "Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR 2017."
    }

    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        split: str = "train",
        train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
        random_seed: int = 42
    ):
        self.image_dir = Path(image_dir)
        self.csv_file = Path(csv_file)
        self.split = split
        self.target_size = target_size
        self.random_seed = random_seed
        self.train_ratio, self.val_ratio, self.test_ratio = train_val_test_ratio

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms(split, target_size)

        self.samples: List[Tuple[Path, List[int], str]] = []  # (path, multi_hot_labels, patient_id)
        if self.csv_file.exists():
            self._load_and_split_patients()

    def default_transforms(self, split: str, size: Tuple[int, int]) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_and_split_patients(self):
        patient_records: Dict[str, List[dict]] = {}
        all_patient_ids: List[str] = []

        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("Image Index", "")
                findings = row.get("Finding Labels", "")
                patient_id = str(row.get("Patient ID", ""))

                if not patient_id:
                    continue

                if patient_id not in patient_records:
                    patient_records[patient_id] = []
                    all_patient_ids.append(patient_id)

                patient_records[patient_id].append({
                    "image_name": img_name,
                    "findings": findings
                })

        # Deterministic patient-level split based on random_seed
        import random
        rng = random.Random(self.random_seed)
        shuffled_patients = list(all_patient_ids)
        rng.shuffle(shuffled_patients)

        n_total = len(shuffled_patients)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        if self.split == "train":
            assigned_patients = set(shuffled_patients[:n_train])
        elif self.split == "val":
            assigned_patients = set(shuffled_patients[n_train:n_train + n_val])
        else:  # test
            assigned_patients = set(shuffled_patients[n_train + n_val:])

        # Build sample list
        for pid in assigned_patients:
            for rec in patient_records[pid]:
                img_path = self.image_dir / rec["image_name"]
                # In NIH dataset, images can be in subfolders or directly in image_dir
                if not img_path.exists():
                    found_paths = list(self.image_dir.glob(f"**/{rec['image_name']}"))
                    if found_paths:
                        img_path = found_paths[0]
                    else:
                        continue

                label_vector = [0] * len(NIH_14_CLASSES)
                findings_list = [f.strip() for f in rec["findings"].split("|")]
                for f in findings_list:
                    if f in CLASS_TO_IDX:
                        label_vector[CLASS_TO_IDX[f]] = 1

                self.samples.append((img_path, label_vector, pid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_vector, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_vector, dtype=torch.float32)

    def get_class_frequencies(self) -> Dict[str, int]:
        counts = {cls: 0 for cls in NIH_14_CLASSES}
        for _, label_vec, _ in self.samples:
            for idx, val in enumerate(label_vec):
                if val == 1:
                    counts[NIH_14_CLASSES[idx]] += 1
        return counts
