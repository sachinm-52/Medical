"""
Evaluation Module for MedVision AI.
Calculates rigorous classification metrics:
- ROC-AUC (per-class and macro-average)
- PR-AUC (Average Precision per-class and macro-average)
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
For binary and multi-label datasets without fabricated results.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    task_type: str = "multi_label",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run evaluation loop over dataloader and compute metrics.
    Returns dictionary with overall and per-class metrics.
    """
    model.eval()
    all_targets = []
    all_preds_prob = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds_prob.append(probs.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_preds_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "task_type": task_type,
        "num_samples": int(len(y_true)),
        "class_names": class_names,
        "per_class": {},
    }

    if task_type == "binary":
        y_true_flat = y_true.ravel()
        y_prob_flat = y_prob.ravel()
        y_pred_flat = y_pred.ravel()

        # Compute ROC-AUC and PR-AUC if both classes present
        try:
            roc_auc = float(roc_auc_score(y_true_flat, y_prob_flat))
        except Exception:
            roc_auc = 0.0

        try:
            pr_auc = float(average_precision_score(y_true_flat, y_prob_flat))
        except Exception:
            pr_auc = 0.0

        acc = float(accuracy_score(y_true_flat, y_pred_flat))
        prec = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
        rec = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))
        f1 = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))
        cm = confusion_matrix(y_true_flat, y_pred_flat).tolist()

        metrics["accuracy"] = acc
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc
        metrics["precision"] = prec
        metrics["recall"] = rec
        metrics["f1_score"] = f1
        metrics["confusion_matrix"] = cm

    else:  # Multi-label
        # Per-class calculations
        num_classes = len(class_names)
        per_class_roc = []
        per_class_pr = []
        per_class_f1 = []
        per_class_prec = []
        per_class_rec = []

        for i in range(num_classes):
            c_true = y_true[:, i]
            c_prob = y_prob[:, i]
            c_pred = y_pred[:, i]
            name = class_names[i]

            # Compute AUC if positive and negative examples exist
            if len(np.unique(c_true)) > 1:
                try:
                    c_roc = float(roc_auc_score(c_true, c_prob))
                except Exception:
                    c_roc = 0.5
                try:
                    c_pr = float(average_precision_score(c_true, c_prob))
                except Exception:
                    c_pr = 0.0
            else:
                c_roc = 0.5
                c_pr = 0.0

            c_prec = float(precision_score(c_true, c_pred, zero_division=0))
            c_rec = float(recall_score(c_true, c_pred, zero_division=0))
            c_f1 = float(f1_score(c_true, c_pred, zero_division=0))

            per_class_roc.append(c_roc)
            per_class_pr.append(c_pr)
            per_class_f1.append(c_f1)
            per_class_prec.append(c_prec)
            per_class_rec.append(c_rec)

            metrics["per_class"][name] = {
                "roc_auc": round(c_roc, 4),
                "pr_auc": round(c_pr, 4),
                "precision": round(c_prec, 4),
                "recall": round(c_rec, 4),
                "f1_score": round(c_f1, 4),
                "support": int(np.sum(c_true))
            }

        # Macro averages
        metrics["roc_auc"] = round(float(np.mean(per_class_roc)), 4)
        metrics["pr_auc"] = round(float(np.mean(per_class_pr)), 4)
        metrics["f1_score"] = round(float(np.mean(per_class_f1)), 4)
        metrics["precision"] = round(float(np.mean(per_class_prec)), 4)
        metrics["recall"] = round(float(np.mean(per_class_rec)), 4)
        # Subset exact accuracy
        metrics["accuracy"] = round(float(np.mean(np.all(y_true == y_pred, axis=1))), 4)

    return metrics
