"""
End-to-End Training Pipeline for MedVision AI.
Supports all 4 datasets:
1. Chest X-ray Pneumonia (Binary)
2. NIH ChestX-ray14 (Multi-label)
3. CheXpert (Multi-label with uncertainty)
4. MedMNIST ChestMNIST (Multi-label lightweight)

Supports architectures: DenseNet121, ResNet50, EfficientNet-B0.
Supports CPU, CUDA GPU, and Google Colab GPU.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets import (
    ChestXrayPneumoniaDataset, PNEUMONIA_CLASSES,
    NIHChestXray14Dataset, NIH_14_CLASSES,
    CheXpertDataset, CHEXPERT_14_CLASSES,
    ChestMNISTDataset, CHESTMNIST_CLASSES
)
from ml.architectures.factory import create_model, ARCHITECTURE_REGISTRY
from ml.evaluation.evaluate import evaluate_model

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Masked BCEWithLogitsLoss for CheXpert when uncertainty_policy='u-ignore' (-1.0).
    Ignores uncertain entries during gradient backpropagation.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = (targets != -1.0).float()
        clean_targets = torch.clamp(targets, min=0.0, max=1.0)
        loss = self.bce(logits, clean_targets)
        loss = (loss * mask).sum() / (mask.sum() + 1e-7)
        return loss

def get_dataset_and_loaders(args):
    """
    Instantiate datasets and DataLoaders according to args.
    """
    dataset_name = args.dataset.lower()

    if dataset_name == "chestmnist":
        train_ds = ChestMNISTDataset(root_dir=args.data_dir, split="train", download=True)
        val_ds = ChestMNISTDataset(root_dir=args.data_dir, split="val", download=True)
        test_ds = ChestMNISTDataset(root_dir=args.data_dir, split="test", download=True)
        classes = CHESTMNIST_CLASSES
        task_type = "multi_label"

    elif dataset_name == "pneumonia":
        train_ds = ChestXrayPneumoniaDataset(root_dir=args.data_dir, split="train")
        val_ds = ChestXrayPneumoniaDataset(root_dir=args.data_dir, split="val")
        test_ds = ChestXrayPneumoniaDataset(root_dir=args.data_dir, split="test")
        classes = PNEUMONIA_CLASSES
        task_type = "binary"

    elif dataset_name == "nih":
        csv_path = Path(args.data_dir) / "Data_Entry_2017.csv"
        img_dir = Path(args.data_dir) / "images"
        train_ds = NIHChestXray14Dataset(image_dir=str(img_dir), csv_file=str(csv_path), split="train")
        val_ds = NIHChestXray14Dataset(image_dir=str(img_dir), csv_file=str(csv_path), split="val")
        test_ds = NIHChestXray14Dataset(image_dir=str(img_dir), csv_file=str(csv_path), split="test")
        classes = NIH_14_CLASSES
        task_type = "multi_label"

    elif dataset_name == "chexpert":
        csv_path = Path(args.data_dir) / "train.csv"
        train_ds = CheXpertDataset(
            root_dir=args.data_dir,
            csv_file=str(csv_path),
            split="train",
            uncertainty_policy=args.uncertainty_policy
        )
        val_csv = Path(args.data_dir) / "valid.csv"
        val_ds = CheXpertDataset(
            root_dir=args.data_dir,
            csv_file=str(val_csv),
            split="val",
            uncertainty_policy=args.uncertainty_policy
        )
        test_ds = val_ds  # CheXpert official validation acts as local test set
        classes = CHEXPERT_14_CLASSES
        task_type = "multi_label"

    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: chestmnist, pneumonia, nih, chexpert")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_loader, val_loader, test_loader, classes, task_type

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"\n==================================================")
    print(f" MedVision AI Training Pipeline")
    print(f" Dataset:      {args.dataset}")
    print(f" Architecture: {args.architecture}")
    print(f" Device:       {device} (CUDA: {torch.cuda.is_available()})")
    print(f" Pretrained:   {args.pretrained} (ImageNet: clearly disclosed)")
    print(f"==================================================\n")

    # Load data
    train_loader, val_loader, test_loader, class_names, task_type = get_dataset_and_loaders(args)
    print(f"Loaded {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples, {len(test_loader.dataset)} test samples.")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Build model
    num_classes = len(class_names)
    model = create_model(
        architecture=args.architecture,
        num_classes=num_classes,
        task_type=task_type,
        pretrained=args.pretrained,
        device=device
    )

    # Loss function
    if task_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        if args.dataset.lower() == "chexpert" and args.uncertainty_policy == "u-ignore":
            criterion = MaskedBCEWithLogitsLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    checkpoint_dir = Path(args.checkpoint_dir) / f"{args.dataset}_{args.architecture.lower()}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": []
    }

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Training loop
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / max(1, n_batches)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        scheduler.step(avg_val_loss)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["epochs"].append(epoch)

        print(f"--> Epoch [{epoch}/{args.epochs}] Done | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "architecture": args.architecture,
                "dataset": args.dataset,
                "task_type": task_type,
                "class_names": class_names,
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "pretrained": args.pretrained
            }, str(best_ckpt_path))
            print(f"  [+] Saved new best model checkpoint to {best_ckpt_path}")

    # Save final model
    final_ckpt_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "architecture": args.architecture,
        "dataset": args.dataset,
        "task_type": task_type,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "history": history
    }, str(final_ckpt_path))

    # Evaluate on test set
    print("\nRunning final evaluation on test set...")
    # Load best weights for test evaluation
    best_ckpt = torch.load(str(checkpoint_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        task_type=task_type
    )
    test_metrics["history"] = history
    test_metrics["training_time_seconds"] = round(time.time() - start_time, 2)
    test_metrics["architecture"] = args.architecture
    test_metrics["dataset"] = args.dataset

    metrics_file = checkpoint_dir / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"ROC-AUC (Macro): {test_metrics.get('roc_auc', 'N/A')}")
    print(f"PR-AUC (Macro):  {test_metrics.get('pr_auc', 'N/A')}")
    print(f"F1 Score:        {test_metrics.get('f1_score', 'N/A')}")
    print(f"Accuracy:        {test_metrics.get('accuracy', 'N/A')}")
    print(f"Metrics saved to: {metrics_file}")
    print("====================================================\n")

    return test_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="MedVision AI Model Training")
    parser.add_argument("--dataset", type=str, default="chestmnist",
                        choices=["chestmnist", "pneumonia", "nih", "chexpert"],
                        help="Dataset to train on")
    parser.add_argument("--architecture", type=str, default="DenseNet121",
                        choices=["DenseNet121", "ResNet50", "EfficientNet-B0"],
                        help="Backbone architecture")
    parser.add_argument("--data_dir", type=str, default="./data/chestmnist",
                        help="Path to dataset root folder or data directory")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker count (0 for Windows compatibility)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet weights (disclosed in UI)")
    parser.add_argument("--uncertainty_policy", type=str, default="u-zeros",
                        choices=["u-zeros", "u-ones", "u-ignore"],
                        help="CheXpert uncertainty policy")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument("--checkpoint_dir", type=str, default="./ml/checkpoints",
                        help="Directory to save checkpoints")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
