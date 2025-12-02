#!/usr/bin/env python3
"""Evaluate trained model on IRMAS test set with multiple pooling strategies.

The IRMAS test set contains polyphonic, variable-length recordings (5-20 seconds)
with multiple instruments but only ONE primary instrument label. We segment each
test file into overlapping windows, run inference on each segment, then aggregate
predictions using different pooling strategies.

Pooling Strategies:
    - Max Pooling: Take maximum probability across all segments
    - Mean Pooling: Average probabilities across all segments  
    - Majority Voting: Most frequent predicted class across segments
    - Confidence Weighted: Weight predictions by confidence (max probability)

Usage:
    python scripts/evaluate_test_set.py \
        --checkpoint outputs/runs/full_augment_*/continued/best_model.pth \
        --test-dir data/raw/IRMAS-TestingData-Part1/Part1 \
        --output-dir outputs/test_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import librosa
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.core.config import load_config
from src.models.densenet import create_densenet121
from src.utils.device import select_device

# IRMAS instrument classes
INSTRUMENTS = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]
INSTRUMENT_NAMES = [
    "Cello",
    "Clarinet",
    "Flute",
    "Acoustic Guitar",
    "Electric Guitar",
    "Organ",
    "Piano",
    "Saxophone",
    "Trumpet",
    "Violin",
    "Voice",
]


class IRMASTestDataset(Dataset):
    """Dataset for IRMAS test set with windowing for long audio files."""

    def __init__(
        self,
        test_dir: Path,
        window_length: float = 3.0,
        hop_length: float = 1.5,
        target_sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length_fft: int = 512,
    ):
        """Initialize test dataset.

        Args:
            test_dir: Path to IRMAS-TestingData-Part1 directory
            window_length: Length of each window in seconds
            hop_length: Hop length between windows in seconds
            target_sr: Target sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length_fft: FFT hop length
        """
        self.test_dir = Path(test_dir)
        self.window_length = window_length
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length_fft = hop_length_fft

        # Find all test files and parse labels
        self.samples = []
        self._load_test_samples()

        print(f"Loaded {len(self.samples)} test samples")

    def _load_test_samples(self):
        """Load all test samples and parse labels from .txt files."""
        # IRMAS test data can be in subdirectories (Part1, Part2, Part3)
        # Check if files are directly in test_dir or in subdirectories
        wav_files = sorted(self.test_dir.glob("*.wav"))

        if len(wav_files) == 0:
            # Try looking in subdirectories (Part1, Part2, Part3)
            print(f"No WAV files in {self.test_dir}, checking subdirectories...")
            for subdir in sorted(self.test_dir.iterdir()):
                if subdir.is_dir():
                    subdir_wavs = sorted(subdir.glob("*.wav"))
                    if len(subdir_wavs) > 0:
                        print(f"  Found {len(subdir_wavs)} files in {subdir.name}/")
                        wav_files.extend(subdir_wavs)

        if len(wav_files) == 0:
            raise ValueError(f"No WAV files found in {self.test_dir} or subdirectories")

        for wav_file in wav_files:
            # Read label from corresponding .txt file
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                print(f"Warning: No label file for {wav_file.name}")
                continue

            with open(txt_file, "r") as f:
                # Primary instrument is on first line
                lines = f.readlines()
                if len(lines) == 0:
                    print(f"Warning: Empty label file for {wav_file.name}")
                    continue
                primary_instrument = lines[0].strip()

            # Convert instrument name to index
            if primary_instrument in INSTRUMENTS:
                label_idx = INSTRUMENTS.index(primary_instrument)
                self.samples.append(
                    {
                        "wav_path": wav_file,
                        "label": primary_instrument,
                        "label_idx": label_idx,
                    }
                )
            else:
                print(
                    f"Warning: Unknown instrument '{primary_instrument}' in {wav_file.name}"
                )

    def _extract_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        """Extract overlapping windows from audio.

        Args:
            audio: Audio waveform

        Returns:
            List of audio windows
        """
        window_samples = int(self.window_length * self.target_sr)
        hop_samples = int(self.hop_length * self.target_sr)

        windows = []
        start = 0

        while start + window_samples <= len(audio):
            window = audio[start : start + window_samples]
            windows.append(window)
            start += hop_samples

        # Handle last window if audio doesn't divide evenly
        if start < len(audio) and len(audio) - start > window_samples // 2:
            # Pad to window length
            last_window = audio[start:]
            pad_length = window_samples - len(last_window)
            last_window = np.pad(last_window, (0, pad_length), mode="constant")
            windows.append(last_window)

        return windows

    def _audio_to_melspec(self, audio: np.ndarray) -> torch.Tensor:
        """Convert audio to mel-spectrogram.

        CRITICAL: Must match training preprocessing exactly!
        - Keep raw dB scale (no normalization to [0,1])
        - Repeat single channel to 3 channels for DenseNet

        Args:
            audio: Audio waveform

        Returns:
            Mel-spectrogram tensor [3, n_mels, time]
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length_fft,
            n_mels=self.n_mels,
        )

        # Convert to dB scale (same as training)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # DO NOT normalize to [0,1] - keep raw dB scale!
        # Training data is in range [-80, 0] dB

        # Convert to tensor and add channel dimension
        mel_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0)

        # Repeat to 3 channels (DenseNet expects RGB)
        # This matches what happens during training (implicitly or explicitly)
        mel_tensor = mel_tensor.repeat(3, 1, 1)

        return mel_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get test sample with all windows.

        Returns:
            Dict with:
                - windows: List of mel-spectrogram tensors
                - label_idx: Ground truth label index
                - filename: Original filename
        """
        sample = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(
            sample["wav_path"],
            sr=self.target_sr,
            mono=True,
        )

        # Extract windows
        audio_windows = self._extract_windows(audio)

        # Convert each window to mel-spectrogram
        mel_windows = [self._audio_to_melspec(window) for window in audio_windows]

        return {
            "windows": mel_windows,
            "label_idx": sample["label_idx"],
            "filename": sample["wav_path"].name,
        }


def collate_test_batch(batch):
    """Custom collate function for variable-length test samples.

    Since each test sample has different number of windows, we return
    a list of samples rather than batching.
    """
    return batch


def inference_on_sample(
    model: torch.nn.Module,
    windows: List[torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on all windows of a sample.

    Args:
        model: Trained model
        windows: List of mel-spectrogram tensors
        device: Device to run on

    Returns:
        Tuple of (probabilities, predicted_classes) arrays
        - probabilities: [num_windows, num_classes]
        - predicted_classes: [num_windows]
    """
    model.eval()

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for window in windows:
            # Add batch dimension and move to device
            window_batch = window.unsqueeze(0).to(device)

            # Forward pass
            logits = model(window_batch)
            probs = F.softmax(logits, dim=1)

            # Get prediction
            pred_class = torch.argmax(probs, dim=1).item()

            all_probs.append(probs.cpu().numpy()[0])
            all_preds.append(pred_class)

    return np.array(all_probs), np.array(all_preds)


def pool_predictions_max(probabilities: np.ndarray) -> int:
    """Max pooling: Take class with maximum probability across all windows.

    Args:
        probabilities: [num_windows, num_classes]

    Returns:
        Predicted class index
    """
    # Take max probability for each class across all windows
    max_probs = np.max(probabilities, axis=0)
    return np.argmax(max_probs)


def pool_predictions_mean(probabilities: np.ndarray) -> int:
    """Mean pooling: Average probabilities across windows.

    Args:
        probabilities: [num_windows, num_classes]

    Returns:
        Predicted class index
    """
    # Average probabilities across windows
    mean_probs = np.mean(probabilities, axis=0)
    return np.argmax(mean_probs)


def pool_predictions_majority(predicted_classes: np.ndarray) -> int:
    """Majority voting: Most frequent predicted class.

    Args:
        predicted_classes: [num_windows]

    Returns:
        Predicted class index
    """
    # Find most common prediction
    unique, counts = np.unique(predicted_classes, return_counts=True)
    return unique[np.argmax(counts)]


def pool_predictions_confidence_weighted(probabilities: np.ndarray) -> int:
    """Confidence-weighted mean: Weight predictions by their confidence.

    Each window's prediction is weighted by its maximum probability
    (confidence). This gives more weight to clear, confident predictions
    and less to uncertain ones.

    Args:
        probabilities: [num_windows, num_classes]

    Returns:
        Predicted class index
    """
    # Get confidence (max probability) for each window
    confidences = np.max(probabilities, axis=1)  # [num_windows]

    # Weight probabilities by confidence
    weighted_probs = probabilities * confidences[:, np.newaxis]

    # Sum and normalize
    weighted_mean = np.sum(weighted_probs, axis=0) / (np.sum(confidences) + 1e-8)

    return np.argmax(weighted_mean)


def run_inference_all_samples(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Run inference once on all test samples, cache results.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Tuple of:
        - all_probabilities: List of [num_windows, num_classes] arrays
        - all_predictions: List of [num_windows] arrays
        - all_labels: List of ground truth label indices
    """
    all_probabilities = []
    all_predictions = []
    all_labels = []

    print("\nRunning inference on all test samples (once)...")

    # Debug: Check first sample
    debug_sample = True

    for batch in tqdm(test_loader, desc="Inference"):
        for sample in batch:
            windows = sample["windows"]
            label_idx = sample["label_idx"]

            # Debug first sample
            if debug_sample and len(windows) > 0:
                print("\n  DEBUG: First sample")
                print(f"    Filename: {sample['filename']}")
                print(f"    Num windows: {len(windows)}")
                print(f"    Window shape: {windows[0].shape}")
                print(
                    f"    Window stats: min={windows[0].min():.2f}, max={windows[0].max():.2f}, mean={windows[0].mean():.2f}"
                )
                print(f"    Ground truth: {INSTRUMENTS[label_idx]} (idx={label_idx})")
                print("    Expected range: [-80, 0] dB (raw dB scale)")
                debug_sample = False

            # Run inference on all windows (once!)
            probabilities, predicted_classes = inference_on_sample(
                model, windows, device
            )

            all_probabilities.append(probabilities)
            all_predictions.append(predicted_classes)
            all_labels.append(label_idx)

    return all_probabilities, all_predictions, all_labels


def apply_pooling_strategy(
    all_probabilities: List[np.ndarray],
    all_predictions: List[np.ndarray],
    all_labels: List[int],
    pooling_strategy: str,
) -> Tuple[float, List[int]]:
    """Apply pooling strategy to cached inference results.

    Args:
        all_probabilities: List of probability arrays per sample
        all_predictions: List of prediction arrays per sample
        all_labels: Ground truth labels
        pooling_strategy: One of ['max', 'mean', 'majority', 'confidence_weighted']

    Returns:
        Tuple of (f1_score, final_predictions)
    """
    final_predictions = []

    for probabilities, predicted_classes in zip(all_probabilities, all_predictions):
        # Apply pooling strategy
        if pooling_strategy == "max":
            prediction = pool_predictions_max(probabilities)
        elif pooling_strategy == "mean":
            prediction = pool_predictions_mean(probabilities)
        elif pooling_strategy == "majority":
            prediction = pool_predictions_majority(predicted_classes)
        elif pooling_strategy == "confidence_weighted":
            prediction = pool_predictions_confidence_weighted(probabilities)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        final_predictions.append(prediction)

    # Compute F1 score
    f1_macro = f1_score(all_labels, final_predictions, average="macro", zero_division=0)

    return f1_macro, final_predictions


def save_results(
    output_dir: Path,
    results: Dict,
    confusion_matrices: Dict,
    classification_reports: Dict,
):
    """Save evaluation results to disk.

    Args:
        output_dir: Output directory
        results: Dictionary of results by pooling strategy
        confusion_matrices: Dictionary of confusion matrices
        classification_reports: Dictionary of classification reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary results
    summary = {
        "pooling_comparison": {
            strategy: {
                "macro_f1": float(results[strategy]["macro_f1"]),
                "micro_f1": float(results[strategy]["micro_f1"]),
            }
            for strategy in ["max", "mean", "majority", "confidence_weighted"]
        },
        "best_strategy": max(
            ["max", "mean", "majority", "confidence_weighted"],
            key=lambda x: results[x]["macro_f1"],
        ),
    }

    with open(output_dir / "pooling_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results for each strategy
    for strategy in ["max", "mean", "majority", "confidence_weighted"]:
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(exist_ok=True)

        # Save confusion matrix
        cm = confusion_matrices[strategy]
        np.save(strategy_dir / "confusion_matrix.npy", cm)

        # Save as CSV for easy viewing
        cm_df = pd.DataFrame(
            cm,
            index=INSTRUMENT_NAMES,
            columns=INSTRUMENT_NAMES,
        )
        cm_df.to_csv(strategy_dir / "confusion_matrix.csv")

        # Save classification report
        with open(strategy_dir / "classification_report.txt", "w") as f:
            f.write(classification_reports[strategy])

    print(f"\n✓ Results saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model on IRMAS test set with pooling strategies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to IRMAS test directory (will search for .wav files in this dir and subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=3.0,
        help="Window length in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--hop-length",
        type=float,
        default=1.5,
        help="Hop length in seconds (default: 1.5, 50%% overlap)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("IRMAS TEST SET EVALUATION - POOLING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test directory: {args.test_dir}")
    print(f"Window length: {args.window_length}s")
    print(
        f"Hop length: {args.hop_length}s (overlap: {(1 - args.hop_length / args.window_length) * 100:.0f}%)"
    )
    print()

    # Load config
    config = load_config(args.config)
    device = select_device()

    print(f"Device: {device}")
    print()

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = create_densenet121(
        num_classes=11,
        pretrained=False,  # We're loading trained weights
        dropout_rate=config["model"]["dropout"],
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded checkpoint from: {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"✓ Trained for {checkpoint['epoch']} epochs")
    if "val_f1" in checkpoint:
        print(f"✓ Validation F1: {checkpoint['val_f1']:.4f}")
    print()

    # =========================================================================
    # LOAD TEST DATA
    # =========================================================================
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    test_dataset = IRMASTestDataset(
        test_dir=args.test_dir,
        window_length=args.window_length,
        hop_length=args.hop_length,
        target_sr=config["audio"]["sample_rate"],
        n_mels=config["features"]["n_mels"],
        n_fft=config["features"]["n_fft"],
        hop_length_fft=config["features"]["hop_length"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one file at a time
        shuffle=False,
        num_workers=0,  # Must be 0 for custom collate
        collate_fn=collate_test_batch,
    )

    if len(test_dataset) == 0:
        print("❌ No test samples loaded! Check directory structure.")
        print(f"   Expected: {args.test_dir}/*.wav")
        print(f"   Or subdirs: {args.test_dir}/Part*/*.wav")
        sys.exit(1)

    print(f"✓ Test samples: {len(test_dataset)}")
    print()

    # =========================================================================
    # RUN INFERENCE ONCE
    # =========================================================================
    print("=" * 70)
    print("INFERENCE")
    print("=" * 70)

    all_probabilities, all_predictions, all_labels = run_inference_all_samples(
        model, test_loader, device
    )

    print(f"\n✓ Inference complete on {len(all_labels)} samples")

    # =========================================================================
    # APPLY POOLING STRATEGIES
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPLYING POOLING STRATEGIES")
    print("=" * 70)

    results = {}
    confusion_matrices = {}
    classification_reports = {}
    all_final_predictions = {}

    for strategy in ["max", "mean", "majority", "confidence_weighted"]:
        print(f"\nApplying {strategy.replace('_', ' ')} pooling...")

        f1_macro, final_predictions = apply_pooling_strategy(
            all_probabilities, all_predictions, all_labels, strategy
        )

        all_final_predictions[strategy] = final_predictions

        # Compute additional metrics
        f1_micro = f1_score(all_labels, final_predictions, average="micro")

        # Use labels parameter to handle missing classes in test set
        cm = confusion_matrix(all_labels, final_predictions, labels=range(11))
        report = classification_report(
            all_labels,
            final_predictions,
            target_names=INSTRUMENT_NAMES,
            labels=range(11),  # Specify all possible labels
            digits=4,
            zero_division=0,  # Set undefined metrics to 0
        )

        results[strategy] = {
            "macro_f1": f1_macro,
            "micro_f1": f1_micro,
        }
        confusion_matrices[strategy] = cm
        classification_reports[strategy] = report

        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Micro F1: {f1_micro:.4f}")

        # Print prediction distribution for debugging
        pred_counts = np.bincount(final_predictions, minlength=11)
        label_counts = np.bincount(all_labels, minlength=11)
        print(f"  Prediction distribution: {pred_counts}")
        print(f"  Ground truth distribution: {label_counts}")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("POOLING STRATEGY COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(
        {
            "Strategy": [
                "Max Pooling",
                "Mean Pooling",
                "Majority Voting",
                "Confidence Weighted",
            ],
            "Macro F1": [
                results["max"]["macro_f1"],
                results["mean"]["macro_f1"],
                results["majority"]["macro_f1"],
                results["confidence_weighted"]["macro_f1"],
            ],
            "Micro F1": [
                results["max"]["micro_f1"],
                results["mean"]["micro_f1"],
                results["majority"]["micro_f1"],
                results["confidence_weighted"]["micro_f1"],
            ],
        }
    )

    print("\n" + results_df.to_string(index=False))

    best_strategy = max(
        ["max", "mean", "majority", "confidence_weighted"],
        key=lambda x: results[x]["macro_f1"],
    )
    best_display_name = {
        "max": "MAX POOLING",
        "mean": "MEAN POOLING",
        "majority": "MAJORITY VOTING",
        "confidence_weighted": "CONFIDENCE WEIGHTED",
    }
    print(f"\n🏆 Best strategy: {best_display_name[best_strategy]}")
    print(f"   Test Macro F1: {results[best_strategy]['macro_f1']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    save_results(output_dir, results, confusion_matrices, classification_reports)

    # Save comparison table
    results_df.to_csv(output_dir / "pooling_comparison.csv", index=False)

    print("\n✓ Evaluation complete!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
