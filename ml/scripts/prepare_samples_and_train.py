"""
MedVision AI Sample Extractor & Model Trainer.
Downloads ChestMNIST (MedMNIST subset), saves sample test X-rays for instant UI testing,
trains real DenseNet121 and ResNet50 models, and generates genuine evaluation metrics.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.chestmnist import ChestMNISTDataset, CHESTMNIST_CLASSES
from ml.architectures.factory import create_model
from ml.evaluation.evaluate import evaluate_model

SAMPLE_DIR = PROJECT_ROOT / "ml" / "samples"
PUBLIC_SAMPLE_DIR = PROJECT_ROOT / "frontend" / "public" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def extract_sample_xrays():
    """Extract sample chest X-rays from ChestMNIST dataset and save as standard images"""
    print("Loading ChestMNIST test set to extract reference X-ray images...")
    dataset = ChestMNISTDataset(root_dir=str(PROJECT_ROOT / "data" / "chestmnist"), split="test", download=True)

    # Find specific conditions: Normal (all zeros), Cardiomegaly, Pneumonia, Effusion, Infiltration
    samples_found = {
        "normal": None,
        "cardiomegaly": None,
        "pneumonia": None,
        "effusion": None,
        "multidisease": None
    }

    cardiomegaly_idx = CHESTMNIST_CLASSES.index("Cardiomegaly")
    pneumonia_idx = CHESTMNIST_CLASSES.index("Pneumonia")
    effusion_idx = CHESTMNIST_CLASSES.index("Effusion")

    for i in range(len(dataset)):
        raw_img = dataset.images[i]  # 28x28 numpy array
        labels = dataset.labels[i]   # 14 binary indicators

        # Upscale 28x28 to clear 224x224 grayscale radiograph for presentation
        img_pil = Image.fromarray(raw_img).resize((224, 224), Image.Resampling.LANCZOS)

        # Normal: all 0
        if samples_found["normal"] is None and np.sum(labels) == 0:
            samples_found["normal"] = (img_pil, "sample_normal.png", "Normal Patient Chest Radiograph")

        # Cardiomegaly
        if samples_found["cardiomegaly"] is None and labels[cardiomegaly_idx] == 1:
            samples_found["cardiomegaly"] = (img_pil, "sample_cardiomegaly.png", "Cardiomegaly Finding")

        # Pneumonia
        if samples_found["pneumonia"] is None and labels[pneumonia_idx] == 1:
            samples_found["pneumonia"] = (img_pil, "sample_pneumonia.png", "Pneumonia Finding")

        # Effusion
        if samples_found["effusion"] is None and labels[effusion_idx] == 1:
            samples_found["effusion"] = (img_pil, "sample_effusion.png", "Pleural Effusion Finding")

        # Multi-disease: 3 or more findings
        if samples_found["multidisease"] is None and np.sum(labels) >= 3:
            samples_found["multidisease"] = (img_pil, "sample_multidisease.png", "Multi-Pathology Presentation")

        if all(v is not None for v in samples_found.values()):
            break

    # Save images
    manifest = []
    for key, (img, filename, label_desc) in samples_found.items():
        save_path = SAMPLE_DIR / filename
        img.save(save_path)
        img.save(PUBLIC_SAMPLE_DIR / filename)
        manifest.append({
            "key": key,
            "filename": filename,
            "label_desc": label_desc,
            "url": f"/api/samples/{filename}"
        })
        print(f"  [+] Saved {filename} -> {save_path}")

    with open(SAMPLE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return dataset

def train_and_evaluate_models(dataset):
    """
    Train and save real weights and genuine evaluation metrics.
    """
    device = torch.device("cpu")
    print(f"\nTraining MedVision AI DenseNet121 Model on ChestMNIST...")

    # We will use a representative subset of train images for fast convergence
    train_ds = ChestMNISTDataset(root_dir=str(PROJECT_ROOT / "data" / "chestmnist"), split="train", download=False)
    val_ds = ChestMNISTDataset(root_dir=str(PROJECT_ROOT / "data" / "chestmnist"), split="val", download=False)
    test_ds = dataset

    # Create subset for training
    torch.manual_seed(42)
    train_subset_indices = torch.randperm(len(train_ds))[:2000]
    val_subset_indices = torch.randperm(len(val_ds))[:500]
    test_subset_indices = torch.randperm(len(test_ds))[:1000]

    train_sub = torch.utils.data.Subset(train_ds, train_subset_indices)
    val_sub = torch.utils.data.Subset(val_ds, val_subset_indices)
    test_sub = torch.utils.data.Subset(test_ds, test_subset_indices)

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_sub, batch_size=32, shuffle=False)

    for arch in ["DenseNet121", "ResNet50", "EfficientNet-B0"]:
        print(f"\n---> Training Architecture: {arch}")
        model = create_model(
            architecture=arch,
            num_classes=14,
            task_type="multi_label",
            pretrained=False,
            device=device
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        epochs = 2
        history = {"train_loss": [], "val_loss": [], "epochs": []}

        for ep in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)

            model.eval()
            v_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    out = model(imgs)
                    loss = criterion(out, targets)
                    v_loss += loss.item()
                    v_batches += 1
            val_loss = v_loss / max(1, v_batches)

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["epochs"].append(ep)
            print(f"  Epoch [{ep}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Checkpoint directory
        slug = f"chestmnist_{arch.lower()}"
        ckpt_dir = PROJECT_ROOT / "ml" / "checkpoints" / slug
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best_model.pth"

        torch.save({
            "architecture": arch,
            "dataset": "MedMNIST (ChestMNIST)",
            "task_type": "multi_label",
            "class_names": CHESTMNIST_CLASSES,
            "state_dict": model.state_dict(),
            "history": history
        }, str(best_path))
        print(f"  [+] Saved trained weights to: {best_path}")

        # Evaluate on test set
        print(f"  Evaluating {arch} on ChestMNIST test split...")
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=CHESTMNIST_CLASSES,
            task_type="multi_label"
        )
        metrics["history"] = history
        metrics["architecture"] = arch
        metrics["dataset"] = "MedMNIST (ChestMNIST)"

        metrics_path = ckpt_dir / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [+] Saved real experimental metrics to: {metrics_path}")
        print(f"      ROC-AUC (Macro): {metrics.get('roc_auc')} | F1: {metrics.get('f1_score')}")

if __name__ == "__main__":
    test_ds = extract_sample_xrays()
    train_and_evaluate_models(test_ds)
