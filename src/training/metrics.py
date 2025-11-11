"""Training metrics for musical instrument classification.

Computes accuracy, macro F1-score, and per-class metrics for evaluation.

Example:
    >>> from src.training.metrics import compute_metrics
    >>>
    >>> # Get predictions from model
    >>> logits = model(batch)
    >>> preds = torch.argmax(logits, dim=1)
    >>>
    >>> # Compute metrics
    >>> metrics = compute_metrics(preds.cpu(), labels.cpu(), num_classes=11)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    >>> print(f"Macro F1: {metrics['macro_f1']:.4f}")
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
        Dictionary with metrics:
        - accuracy: Overall accuracy
        - macro_f1: Macro-averaged F1-score (target metric for IRMAS)
        - weighted_f1: Weighted F1-score
        - macro_precision: Macro-averaged precision
        - macro_recall: Macro-averaged recall

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1, 0])
        >>> targets = torch.tensor([0, 1, 1, 1, 0])
        >>> metrics = compute_metrics(preds, targets, num_classes=3)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        Accuracy: 0.8000
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
    """Compute per-class metrics.

    Args:
        predictions: Predicted class indices.
        targets: Ground truth class indices.
        class_names: List of class names (e.g., IRMAS_CLASSES).

    Returns:
        Dictionary mapping class names to their metrics (F1, precision, recall).

    Example:
        >>> from src.data.irmas import IRMAS_CLASSES
        >>> per_class = compute_per_class_metrics(preds, targets, IRMAS_CLASSES)
        >>> print(f"Piano F1: {per_class['pia']['f1']:.4f}")
    """
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
    """Compute confusion matrix.

    Args:
        predictions: Predicted class indices.
        targets: Ground truth class indices.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
        Entry [i, j] is the number of samples with true label i predicted as j.

    Example:
        >>> cm = get_confusion_matrix(preds, targets, num_classes=11)
        >>> print(cm.shape)
        (11, 11)
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    return cm


class MetricsTracker:
    """Track metrics during training and validation.

    Accumulates predictions and targets across batches, then computes
    metrics at the end of an epoch.

    Attributes:
        predictions: List of prediction tensors.
        targets: List of target tensors.
        num_classes: Number of classes.

    Example:
        >>> tracker = MetricsTracker(num_classes=11)
        >>>
        >>> # During epoch
        >>> for batch in dataloader:
        ...     logits = model(batch)
        ...     preds = torch.argmax(logits, dim=1)
        ...     tracker.update(preds, labels)
        >>>
        >>> # At end of epoch
        >>> metrics = tracker.compute()
        >>> print(f"Epoch accuracy: {metrics['accuracy']:.4f}")
        >>> tracker.reset()
    """

    def __init__(self, num_classes: int = 11):
        """Initialize metrics tracker.

        Args:
            num_classes: Number of classes (default: 11).
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Add batch predictions and targets.

        Args:
            predictions: Predicted class indices for batch.
            targets: Ground truth class indices for batch.
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute metrics from accumulated predictions and targets.

        Returns:
            Dictionary with computed metrics.
        """
        # Concatenate all batches
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, self.num_classes)

        return metrics

    def compute_per_class(self, class_names: list[str]) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics.

        Args:
            class_names: List of class names.

        Returns:
            Per-class metrics dictionary.
        """
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        return compute_per_class_metrics(all_preds, all_targets, class_names)
