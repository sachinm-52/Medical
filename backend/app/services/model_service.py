"""
Model Registry Service for MedVision AI.
Manages metadata, provenance, and availability of models across the four datasets.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.schema_models import ModelRecord, ModelMetric
from backend.app.config import settings
from ml.datasets import PNEUMONIA_CLASSES, NIH_14_CLASSES, CHEXPERT_14_CLASSES, CHESTMNIST_CLASSES

DEFAULT_MODELS = [
    {
        "model_name": "ChestMNIST Multi-Disease Model (DenseNet121)",
        "architecture": "DenseNet121",
        "dataset_name": "MedMNIST (ChestMNIST)",
        "model_version": "1.0.0",
        "task_type": "multi_label",
        "num_classes": 14,
        "class_labels": CHESTMNIST_CLASSES,
        "weights_rel_path": "chestmnist_densenet121/best_model.pth"
    },
    {
        "model_name": "NIH ChestX-ray14 Multi-Disease Model (DenseNet121)",
        "architecture": "DenseNet121",
        "dataset_name": "NIH ChestX-ray14",
        "model_version": "1.0.0",
        "task_type": "multi_label",
        "num_classes": 14,
        "class_labels": NIH_14_CLASSES,
        "weights_rel_path": "nih_densenet121/best_model.pth"
    },
    {
        "model_name": "CheXpert Classification Model (DenseNet121)",
        "architecture": "DenseNet121",
        "dataset_name": "CheXpert",
        "model_version": "1.0.0",
        "task_type": "multi_label",
        "num_classes": 14,
        "class_labels": CHEXPERT_14_CLASSES,
        "weights_rel_path": "chexpert_densenet121/best_model.pth"
    },
    {
        "model_name": "Pneumonia Classification Model (DenseNet121)",
        "architecture": "DenseNet121",
        "dataset_name": "Chest X-ray Pneumonia Dataset",
        "model_version": "1.0.0",
        "task_type": "binary",
        "num_classes": 2,
        "class_labels": PNEUMONIA_CLASSES,
        "weights_rel_path": "pneumonia_densenet121/best_model.pth"
    },
    {
        "model_name": "ChestMNIST Benchmark (ResNet50)",
        "architecture": "ResNet50",
        "dataset_name": "MedMNIST (ChestMNIST)",
        "model_version": "1.0.0",
        "task_type": "multi_label",
        "num_classes": 14,
        "class_labels": CHESTMNIST_CLASSES,
        "weights_rel_path": "chestmnist_resnet50/best_model.pth"
    },
    {
        "model_name": "ChestMNIST Efficient (EfficientNet-B0)",
        "architecture": "EfficientNet-B0",
        "dataset_name": "MedMNIST (ChestMNIST)",
        "model_version": "1.0.0",
        "task_type": "multi_label",
        "num_classes": 14,
        "class_labels": CHESTMNIST_CLASSES,
        "weights_rel_path": "chestmnist_efficientnet-b0/best_model.pth"
    }
]

DATASET_SPECS = [
    {
        "name": "MedMNIST v2 (ChestMNIST)",
        "slug": "chestmnist",
        "description": "Lightweight chest X-ray multi-label benchmark with 14 disease classes based on NIH ChestX-ray14. Natively patient-partitioned without data leakage.",
        "source_url": "https://medmnist.com/",
        "task_type": "Multi-label Classification (14 classes)",
        "num_classes": 14,
        "classes": CHESTMNIST_CLASSES,
        "provenance": "Zenodo Open Access Dataset. 112,120 total images: 78,468 train, 11,219 val, 22,433 test. Pre-split at patient level.",
        "patient_level_split": True,
        "total_images_note": "112,120 standardized chest radiograph patches.",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "uncertainty_handling": "Not applicable (binary multi-label 0/1)"
    },
    {
        "name": "Chest X-ray Pneumonia Dataset",
        "slug": "pneumonia",
        "description": "Pediatric chest radiograph dataset for binary classification of Normal versus Pneumonia (bacterial and viral).",
        "source_url": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "task_type": "Binary Classification (Normal vs Pneumonia)",
        "num_classes": 2,
        "classes": PNEUMONIA_CLASSES,
        "provenance": "Guangzhou Women and Children's Medical Center. Validated by two expert physicians. Split into train, val, and test cohorts.",
        "patient_level_split": True,
        "total_images_note": "5,863 anterior-posterior chest X-ray images.",
        "license": "CC BY 4.0",
        "uncertainty_handling": "Not applicable (binary definitive labels)"
    },
    {
        "name": "NIH ChestX-ray14",
        "slug": "nih_chestxray14",
        "description": "Hospital-scale multi-label chest radiograph dataset detecting 14 common thoracic conditions with patient identifiers for leakage-free splitting.",
        "source_url": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "task_type": "Multi-label Classification (14 diseases)",
        "num_classes": 14,
        "classes": NIH_14_CLASSES,
        "provenance": "National Institutes of Health Clinical Center. 30,805 unique patients. Split deterministically on Patient ID to prevent clinical bias.",
        "patient_level_split": True,
        "total_images_note": "112,120 frontal-view X-ray images.",
        "license": "NIH Open Access Research Use",
        "uncertainty_handling": "Text-mined radiological reports; multi-hot encoded"
    },
    {
        "name": "Stanford CheXpert",
        "slug": "chexpert",
        "description": "Large chest radiograph dataset with explicit uncertainty labels (-1) extracted via automated rule-based NLP from radiology reports.",
        "source_url": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "task_type": "Multi-label Classification with Uncertainty",
        "num_classes": 14,
        "classes": CHEXPERT_14_CLASSES,
        "provenance": "Stanford Hospital radiology studies from 65,240 patients. Grouped by patient ID for leak-free evaluation.",
        "patient_level_split": True,
        "total_images_note": "224,316 chest radiographs.",
        "license": "Stanford University Research Use Agreement",
        "uncertainty_handling": "Explicit support for U-Zeros (conservative), U-Ones (sensitive), and U-Ignore (masked gradient backpropagation)."
    }
]

class ModelService:
    def sync_models(self, db: Session):
        """
        Sync database models with file system checkpoints.
        Only marks models as 'ready' if their weights actually exist.
        """
        for def_model in DEFAULT_MODELS:
            weights_file = settings.CHECKPOINT_DIR / def_model["weights_rel_path"]
            exists = weights_file.exists()
            status = "ready" if exists else "not_trained"

            rec = db.query(ModelRecord).filter_by(model_name=def_model["model_name"]).first()
            if not rec:
                rec = ModelRecord(
                    model_name=def_model["model_name"],
                    architecture=def_model["architecture"],
                    dataset_name=def_model["dataset_name"],
                    model_version=def_model["model_version"],
                    weights_path=str(weights_file) if exists else None,
                    status=status,
                    task_type=def_model["task_type"],
                    num_classes=def_model["num_classes"],
                    class_labels=def_model["class_labels"]
                )
                db.add(rec)
            else:
                rec.status = status
                rec.weights_path = str(weights_file) if exists else None
        db.commit()

    def get_models(self, db: Session, ready_only: bool = False) -> List[ModelRecord]:
        self.sync_models(db)
        query = db.query(ModelRecord)
        if ready_only:
            query = query.filter(ModelRecord.status == "ready")
        return query.all()

    def get_model_by_id(self, db: Session, model_id: int) -> Optional[ModelRecord]:
        self.sync_models(db)
        return db.query(ModelRecord).filter(ModelRecord.id == model_id).first()

    def get_datasets(self) -> List[Dict[str, Any]]:
        return DATASET_SPECS

model_service = ModelService()
