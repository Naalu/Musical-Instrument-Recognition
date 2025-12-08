# scripts/combine_test_results.py
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

# IRMAS instrument names
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

parts = {
    "Part1": "outputs/test_results/fixed",
    "Part2": "outputs/test_results/part2",
    "Part3": "outputs/test_results/part3",
}

print("=" * 70)
print("COMBINING CONFUSION MATRICES FROM ALL PARTS")
print("=" * 70)

# Load and sum confusion matrices
combined_cm = None
part_results = {}

for part_name, part_dir in parts.items():
    # Load confusion matrix for best pooling strategy (confidence_weighted)
    cm_file = Path(part_dir) / "confidence_weighted" / "confusion_matrix.npy"
    cm = np.load(cm_file)

    # Sum confusion matrices
    if combined_cm is None:
        combined_cm = cm
    else:
        combined_cm += cm

    # Compute metrics for this part
    y_true = []
    y_pred = []
    for true_idx in range(len(cm)):
        for pred_idx in range(len(cm)):
            count = int(cm[true_idx, pred_idx])
            y_true.extend([true_idx] * count)
            y_pred.extend([pred_idx] * count)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    part_results[part_name] = {
        "samples": len(y_true),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }

    print(f"\n{part_name}:")
    print(f"  Samples: {len(y_true)}")
    print(f"  Macro F1: {macro_f1:.4f} ({macro_f1 * 100:.1f}%)")
    print(f"  Micro F1: {micro_f1:.4f} ({micro_f1 * 100:.1f}%)")

# Compute combined metrics
print("\n" + "=" * 70)
print("COMBINED RESULTS (All 3 parts)")
print("=" * 70)

y_true_all = []
y_pred_all = []
for true_idx in range(len(combined_cm)):
    for pred_idx in range(len(combined_cm)):
        count = int(combined_cm[true_idx, pred_idx])
        y_true_all.extend([true_idx] * count)
        y_pred_all.extend([pred_idx] * count)

combined_macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
combined_micro_f1 = f1_score(y_true_all, y_pred_all, average="micro")

print(f"\nTotal samples: {len(y_true_all)}")
print(f"Macro F1: {combined_macro_f1:.4f} ({combined_macro_f1 * 100:.1f}%)")
print(f"Micro F1: {combined_micro_f1:.4f} ({combined_micro_f1 * 100:.1f}%)")

print(f"\n{'=' * 70}")
print("COMPARISON TO HAN ET AL. (2017)")
print("=" * 70)
print("Han et al. Micro F1: 0.619 (61.9%)")
print(f"Ours:                {combined_micro_f1:.4f} ({combined_micro_f1 * 100:.1f}%)")
print(
    f"Difference:          {(combined_micro_f1 - 0.619):.4f} ({(combined_micro_f1 - 0.619) * 100:.1f} points)"
)
