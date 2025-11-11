"""Training metrics for musical instrument classification.

Computes accuracy, macro F1-score, and per-class metrics for evaluation.
"""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int = 11,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted class indices, shape (n_samples,).
        targets: Ground truth class indices, shape (n_samples,).
        num_classes: Number of classes (default: 11).
        average: Averaging method for F1/precision/recall (default: 'macro').

    Returns:
        Dictionary with metrics.
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(targets, predictions)

    # F1 scores
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, predictions, average="weighted", zero_division=0)

    # Precision and recall
    macro_precision = precision_score(
        targets, predictions, average="macro", zero_division=0
    )
    macro_recall = recall_score(targets, predictions, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
    }

    return metrics


def compute_per_class_metrics(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    class_names: list[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics."""
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Compute per-class metrics
    f1_scores = f1_score(targets, predictions, average=None, zero_division=0)
    precision_scores = precision_score(
        targets, predictions, average=None, zero_division=0
    )
    recall_scores = recall_score(targets, predictions, average=None, zero_division=0)

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "f1": float(f1_scores[i]),
            "precision": float(precision_scores[i]),
            "recall": float(recall_scores[i]),
        }

    return per_class_metrics


def get_confusion_matrix(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int = 11,
) -> np.ndarray:
    """Compute confusion matrix."""
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    return cm


class MetricsTracker:
    """Track metrics during training and validation."""

    def __init__(self, num_classes: int = 11):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Add batch predictions and targets."""
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute metrics from accumulated predictions and targets."""
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        metrics = compute_metrics(all_preds, all_targets, self.num_classes)
        return metrics

    def compute_per_class(self, class_names: list[str]) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics."""
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        return compute_per_class_metrics(all_preds, all_targets, class_names)
