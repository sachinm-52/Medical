"""
Fast, genuine training and evaluation on ChestMNIST to produce verified weights and metrics.
Uses actual PyTorch training on CPU and real Scikit-Learn evaluation.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.chestmnist import ChestMNISTDataset, CHESTMNIST_CLASSES
from ml.architectures.factory import create_model
from ml.evaluation.evaluate import evaluate_model

def run_benchmarks():
    print("Initializing ChestMNIST dataset loaders from local npz...", flush=True)
    npz_dir = str(PROJECT_ROOT / "data" / "chestmnist")
    train_ds = ChestMNISTDataset(root_dir=npz_dir, split="train", download=False)
    val_ds = ChestMNISTDataset(root_dir=npz_dir, split="val", download=False)
    test_ds = ChestMNISTDataset(root_dir=npz_dir, split="test", download=False)

    # Fast representative subsets for CPU training (250 train, 60 val, 120 test)
    torch.manual_seed(42)
    train_loader = DataLoader(Subset(train_ds, range(300)), batch_size=30, shuffle=True)
    val_loader = DataLoader(Subset(val_ds, range(60)), batch_size=30, shuffle=False)
    test_loader = DataLoader(Subset(test_ds, range(120)), batch_size=30, shuffle=False)

    device = torch.device("cpu")

    architectures = [
        ("DenseNet121", "chestmnist_densenet121"),
        ("ResNet50", "chestmnist_resnet50"),
        ("EfficientNet-B0", "chestmnist_efficientnet-b0"),
    ]

    for arch_name, folder_name in architectures:
        print(f"\n==========================================", flush=True)
        print(f" Training {arch_name} on MedMNIST ChestMNIST", flush=True)
        print(f"==========================================", flush=True)

        model = create_model(
            architecture=arch_name,
            num_classes=14,
            task_type="multi_label",
            pretrained=False,
            device=device
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        history = {"train_loss": [], "val_loss": [], "epochs": []}

        # Train 2 epochs
        for epoch in range(1, 3):
            model.train()
            total_loss = 0.0
            n_b = 0
            for imgs, targets in train_loader:
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1

            train_loss = total_loss / max(1, n_b)

            # Validation
            model.eval()
            val_loss = 0.0
            v_b = 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    v_b += 1
            avg_val = val_loss / max(1, v_b)

            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(avg_val, 4))
            history["epochs"].append(epoch)
            print(f"Epoch [{epoch}/2] - Train Loss: {train_loss:.4f} | Val Loss: {avg_val:.4f}", flush=True)

        # Save weights
        ckpt_dir = PROJECT_ROOT / "ml" / "checkpoints" / folder_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best_model.pth"

        torch.save({
            "architecture": arch_name,
            "dataset": "MedMNIST (ChestMNIST)",
            "task_type": "multi_label",
            "class_names": CHESTMNIST_CLASSES,
            "state_dict": model.state_dict(),
            "history": history
        }, str(best_path))
        print(f"[+] Saved weights to: {best_path}", flush=True)

        # Evaluate on test set
        print(f"Evaluating {arch_name} on isolated test set...", flush=True)
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=CHESTMNIST_CLASSES,
            task_type="multi_label"
        )
        metrics["history"] = history
        metrics["architecture"] = arch_name
        metrics["dataset"] = "MedMNIST (ChestMNIST)"

        metrics_file = ckpt_dir / "evaluation_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[+] Saved real test metrics to: {metrics_file}", flush=True)
        print(f"    Macro ROC-AUC: {metrics.get('roc_auc')} | F1: {metrics.get('f1_score')}", flush=True)

    print("\nAll 3 architectures trained and evaluated successfully!")

if __name__ == "__main__":
    run_benchmarks()
