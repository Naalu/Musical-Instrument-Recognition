#!/usr/bin/env python3
"""Enhanced IRMAS test evaluation with comprehensive pooling strategies.

This script extends the original evaluate_test_set.py with:
1. Additional pooling strategies (linear softmax, top-k, attention, median, product)
2. Configurable window sizes for comparison
3. Comprehensive results export for presentation

Usage:
    # Run all pooling strategies with default 3s windows
    python scripts/evaluate_test_enhanced.py \
        --checkpoint outputs/runs/full_augment_*/continued/best_model.pth \
        --test-dir data/raw/IRMAS-TestingData-Part1/Part1

    # Compare different window sizes
    python scripts/evaluate_test_enhanced.py --window-length 1.0 --hop-length 0.5
"""

import argparse
import json
import sys
from datetime import datetime
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

from src.models.densenet import create_densenet121
from src.utils.device import select_device

# IRMAS classes
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


# =============================================================================
# POOLING STRATEGIES
# =============================================================================


def pool_max(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Max pooling: class with highest max probability across windows."""
    return int(np.argmax(np.max(probs, axis=0)))


def pool_mean(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Mean pooling: average probabilities across windows."""
    return int(np.argmax(np.mean(probs, axis=0)))


def pool_majority(probs: np.ndarray, preds: np.ndarray) -> int:
    """Majority voting: most frequent predicted class."""
    unique, counts = np.unique(preds, return_counts=True)
    return int(unique[np.argmax(counts)])


def pool_confidence_weighted(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Weight predictions by their confidence (max probability)."""
    confidences = np.max(probs, axis=1)
    weighted = probs * confidences[:, np.newaxis]
    weighted_mean = np.sum(weighted, axis=0) / (np.sum(confidences) + 1e-8)
    return int(np.argmax(weighted_mean))


def pool_linear_softmax(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Linear softmax pooling - self-weighted by squared probabilities.
    From Wang et al. (ICASSP 2019) - recommended for polyphonic audio."""
    weighted = (probs**2).sum(axis=0) / (probs.sum(axis=0) + 1e-8)
    return int(np.argmax(weighted))


def pool_topk_mean_3(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Average only top-3 most confident windows per class."""
    k = min(3, probs.shape[0])
    if probs.shape[0] <= k:
        return int(np.argmax(probs.mean(axis=0)))
    top_k = np.partition(probs, -k, axis=0)[-k:]
    return int(np.argmax(top_k.mean(axis=0)))


def pool_topk_mean_5(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Average only top-5 most confident windows per class."""
    k = min(5, probs.shape[0])
    if probs.shape[0] <= k:
        return int(np.argmax(probs.mean(axis=0)))
    top_k = np.partition(probs, -k, axis=0)[-k:]
    return int(np.argmax(top_k.mean(axis=0)))


def pool_attention_weighted(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Weight windows by inverse entropy (low entropy = more confident)."""
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    weights = 1.0 / (entropy + 0.1)  # Add smoothing to avoid div by zero
    weights = weights / weights.sum()
    weighted_mean = np.average(probs, axis=0, weights=weights)
    return int(np.argmax(weighted_mean))


def pool_median(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Median probability per class - robust to outlier windows."""
    return int(np.argmax(np.median(probs, axis=0)))


def pool_geometric_mean(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Geometric mean - requires consistent evidence across windows."""
    log_probs = np.log(probs + 1e-8)
    mean_log = log_probs.mean(axis=0)
    return int(np.argmax(mean_log))


def pool_trimmed_mean(probs: np.ndarray, preds: np.ndarray = None) -> int:
    """Trimmed mean - remove top/bottom 10% before averaging."""
    n = probs.shape[0]
    trim_n = max(1, int(n * 0.1))
    if n <= 2 * trim_n + 1:
        return int(np.argmax(probs.mean(axis=0)))
    sorted_probs = np.sort(probs, axis=0)
    trimmed = sorted_probs[trim_n:-trim_n]
    return int(np.argmax(trimmed.mean(axis=0)))


# Registry of all pooling strategies
POOLING_STRATEGIES = {
    "max": pool_max,
    "mean": pool_mean,
    "majority": pool_majority,
    "confidence_weighted": pool_confidence_weighted,
    "linear_softmax": pool_linear_softmax,
    "topk_mean_3": pool_topk_mean_3,
    "topk_mean_5": pool_topk_mean_5,
    "attention_weighted": pool_attention_weighted,
    "median": pool_median,
    "geometric_mean": pool_geometric_mean,
    "trimmed_mean": pool_trimmed_mean,
}


# =============================================================================
# TEST DATASET
# =============================================================================


class IRMASTestDataset(Dataset):
    """Test dataset with configurable window parameters."""

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
        self.test_dir = Path(test_dir)
        self.window_length = window_length
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length_fft = hop_length_fft
        self.window_samples = int(window_length * target_sr)
        self.hop_samples = int(hop_length * target_sr)
        self.samples = self._index_test_files()
        print(f"Found {len(self.samples)} test samples")

    def _index_test_files(self) -> List[dict]:
        samples = []
        wav_files = list(self.test_dir.rglob("*.wav"))

        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]
                if labels:
                    primary = labels[0]
                    if primary in INSTRUMENTS:
                        samples.append(
                            {
                                "wav_path": wav_path,
                                "label": primary,
                                "label_idx": INSTRUMENTS.index(primary),
                            }
                        )
        return samples

    def _extract_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        windows = []
        start = 0

        while start + self.window_samples <= len(audio):
            windows.append(audio[start : start + self.window_samples])
            start += self.hop_samples

        # Handle case where audio is shorter than window or partial final window
        if len(windows) == 0:
            # Audio shorter than window - pad it
            pad_len = self.window_samples - len(audio)
            windows.append(np.pad(audio, (0, pad_len), mode="constant"))
        elif start < len(audio):
            # Partial final window - pad it
            remaining = audio[start:]
            pad_len = self.window_samples - len(remaining)
            windows.append(np.pad(remaining, (0, pad_len), mode="constant"))

        return windows

    def _audio_to_melspec(self, audio: np.ndarray) -> torch.Tensor:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length_fft,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0)
        return mel_tensor.repeat(3, 1, 1)  # 1->3 channels for DenseNet

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, _ = librosa.load(sample["wav_path"], sr=self.target_sr, mono=True)
        windows = self._extract_windows(audio)
        mel_windows = [self._audio_to_melspec(w) for w in windows]

        return {
            "windows": mel_windows,
            "label_idx": sample["label_idx"],
            "filename": sample["wav_path"].name,
        }


def collate_fn(batch):
    """Custom collate - return list since samples have variable window counts."""
    return batch


# =============================================================================
# INFERENCE
# =============================================================================


def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Run inference once, cache all probabilities for pooling experiments."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    print("\nRunning inference on all test samples...")

    for batch in tqdm(test_loader, desc="Inference"):
        for sample in batch:
            windows = sample["windows"]
            label_idx = sample["label_idx"]

            probs_list = []
            preds_list = []

            with torch.no_grad():
                for window in windows:
                    window_batch = window.unsqueeze(0).to(device)
                    logits = model(window_batch)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    probs_list.append(probs)
                    preds_list.append(np.argmax(probs))

            all_probs.append(np.array(probs_list))
            all_preds.append(np.array(preds_list))
            all_labels.append(label_idx)

    return all_probs, all_preds, all_labels


def evaluate_all_strategies(
    all_probs: List[np.ndarray],
    all_preds: List[np.ndarray],
    all_labels: List[int],
) -> Dict:
    """Evaluate all pooling strategies on cached inference results."""
    results = {}

    print("\nEvaluating pooling strategies...")

    for strategy_name, strategy_fn in POOLING_STRATEGIES.items():
        final_preds = []

        for probs, preds in zip(all_probs, all_preds):
            pred = strategy_fn(probs, preds)
            final_preds.append(pred)

        macro_f1 = f1_score(all_labels, final_preds, average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels, final_preds, average="micro", zero_division=0)

        cm = confusion_matrix(all_labels, final_preds, labels=range(11))
        report = classification_report(
            all_labels,
            final_preds,
            target_names=INSTRUMENT_NAMES,
            labels=range(11),
            digits=4,
            zero_division=0,
        )

        results[strategy_name] = {
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "predictions": final_preds,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        print(
            f"  {strategy_name:22s}: Macro F1 = {macro_f1:.4f}, Micro F1 = {micro_f1:.4f}"
        )

    return results


def save_results(output_dir: Path, results: Dict, config: Dict):
    """Save all results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]["macro_f1"])
    best_f1 = results[best_strategy]["macro_f1"]

    # Save summary JSON
    summary = {
        "config": config,
        "best_strategy": best_strategy,
        "best_macro_f1": best_f1,
        "all_results": {
            k: {"macro_f1": v["macro_f1"], "micro_f1": v["micro_f1"]}
            for k, v in results.items()
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save comparison table as CSV
    rows = [
        {
            "Strategy": k,
            "Macro_F1": f"{v['macro_f1']:.4f}",
            "Micro_F1": f"{v['micro_f1']:.4f}",
        }
        for k, v in sorted(results.items(), key=lambda x: -x[1]["macro_f1"])
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "pooling_comparison.csv", index=False)

    # Save detailed results for each strategy
    for strategy_name, data in results.items():
        strategy_dir = output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)

        np.save(strategy_dir / "confusion_matrix.npy", data["confusion_matrix"])

        cm_df = pd.DataFrame(
            data["confusion_matrix"],
            index=INSTRUMENT_NAMES,
            columns=INSTRUMENT_NAMES,
        )
        cm_df.to_csv(strategy_dir / "confusion_matrix.csv")

        with open(strategy_dir / "classification_report.txt", "w") as f:
            f.write(data["classification_report"])

    print(f"\n✓ Results saved to {output_dir}")
    return best_strategy, best_f1


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced IRMAS test evaluation with multiple pooling strategies"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-dir", type=str, required=True, help="Path to IRMAS test directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test_results_enhanced",
        help="Output directory",
    )
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yml", help="Config file"
    )
    parser.add_argument(
        "--window-length", type=float, default=3.0, help="Window length in seconds"
    )
    parser.add_argument(
        "--hop-length", type=float, default=1.5, help="Hop length in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"win{args.window_length}s_{timestamp}"

    print("=" * 70)
    print("ENHANCED IRMAS TEST EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print(f"Window: {args.window_length}s, Hop: {args.hop_length}s")
    print(f"Output: {output_dir}")
    print()

    # Setup device
    device = select_device()
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = create_densenet121(num_classes=11, pretrained=False, dropout_rate=0.5)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    checkpoint_val_f1 = checkpoint.get("val_f1", "N/A")
    print(f"✓ Model loaded (checkpoint val F1: {checkpoint_val_f1})")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = IRMASTestDataset(
        test_dir=Path(args.test_dir),
        window_length=args.window_length,
        hop_length=args.hop_length,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Run inference once
    all_probs, all_preds, all_labels = run_inference(model, test_loader, device)
    print(f"✓ Inference complete on {len(all_labels)} samples")

    # Evaluate all pooling strategies
    print("\n" + "=" * 70)
    print("POOLING STRATEGY COMPARISON")
    print("=" * 70)

    results = evaluate_all_strategies(all_probs, all_preds, all_labels)

    # Save results
    config = {
        "checkpoint": args.checkpoint,
        "test_dir": args.test_dir,
        "window_length": args.window_length,
        "hop_length": args.hop_length,
        "num_samples": len(all_labels),
    }

    best_strategy, best_f1 = save_results(output_dir, results, config)

    # Print summary
    print("\n" + "=" * 70)
    print(f"🏆 BEST STRATEGY: {best_strategy}")
    print(f"   Macro F1: {best_f1:.4f} ({best_f1 * 100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
