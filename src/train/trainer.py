"""Training loop for musical instrument classification."""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.irmas import IRMAS_CLASSES
from src.metrics.classification import MetricsTracker


class Trainer:
    """Training pipeline for instrument classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        checkpoint_dir: str | Path = "checkpoints",
        patience: int = 10,
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience

        # Tracking
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }

        print("Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Early stopping patience: {patience}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        metrics_tracker = MetricsTracker(num_classes=len(IRMAS_CLASSES))

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (spectrograms, labels) in enumerate(pbar):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            metrics_tracker.update(predictions, labels)

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / len(self.train_loader)
        metrics = metrics_tracker.compute()

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        running_loss = 0.0
        metrics_tracker = MetricsTracker(num_classes=len(IRMAS_CLASSES))

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for spectrograms, labels in pbar:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(spectrograms)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                metrics_tracker.update(predictions, labels)

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / len(self.val_loader)
        metrics = metrics_tracker.compute()

        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_f1": val_f1,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (F1: {val_f1:.4f})")

    def train(self, num_epochs: int, save_every: int = 5) -> Dict[str, list]:
        """Train model for multiple epochs."""
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print("Target metric: Macro F1-score ≥ 0.60")
        print()

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 70)

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["macro_f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["macro_f1"])
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            print(
                f"  Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Train F1: {train_metrics['macro_f1']:.4f}"
            )
            print(
                f"  Val Loss:   {val_metrics['loss']:.4f} | "
                f"Val Acc:   {val_metrics['accuracy']:.4f} | "
                f"Val F1:   {val_metrics['macro_f1']:.4f}"
            )
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            is_best = val_metrics["macro_f1"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.patience_counter = 0
                print(f"  ✨ New best model! (F1: {self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics["macro_f1"], is_best)

            if self.patience_counter >= self.patience:
                print(
                    f"\n⏸️  Early stopping triggered (no improvement for {self.patience} epochs)"
                )
                break

            print()

        total_time = time.time() - start_time
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation F1: {self.best_val_f1:.4f}")

        if self.best_val_f1 >= 0.60:
            print("✅ Target F1-score of 0.60 achieved!")
        else:
            print(f"⚠️  Target F1-score not reached (best: {self.best_val_f1:.4f})")

        return self.history

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")
