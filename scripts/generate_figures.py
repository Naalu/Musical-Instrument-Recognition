"""Generate presentation figures for ablation study."""

import matplotlib.pyplot as plt
import numpy as np

# Data
experiments = ["Baseline", "Pitch", "Stretch", "SpecAug", "Full Aug"]
val_f1 = [59.25, 66.43, 66.40, 62.80, 74.79]
test_f1 = [24.80, 23.98, 25.92, 23.39, 26.77]
train_f1 = [99.55, 88.70, 89.56, 60.92, 70.0]

# Figure 1: Val vs Test F1 Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(experiments))
width = 0.35

bars1 = ax.bar(x - width / 2, val_f1, width, label="Validation F1", color="#2ecc71")
bars2 = ax.bar(x + width / 2, test_f1, width, label="Test F1", color="#e74c3c")

ax.set_ylabel("Macro F1 Score (%)", fontsize=12)
ax.set_title("Augmentation Ablation: Validation vs Test Performance", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()
ax.set_ylim(0, 85)

# Add value labels
for bar in bars1:
    ax.annotate(
        f"{bar.get_height():.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    ax.annotate(
        f"{bar.get_height():.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("outputs/figures/ablation_comparison.png", dpi=150)
plt.savefig("outputs/figures/ablation_comparison.pdf")
print("✓ Saved ablation_comparison.png/pdf")

# Figure 2: Overfitting Analysis (Train-Val Gap)
fig, ax = plt.subplots(figsize=(10, 6))
train_val_gap = [t - v for t, v in zip(train_f1, val_f1)]
colors = [
    "#e74c3c" if g > 10 else "#f39c12" if g > 0 else "#2ecc71" for g in train_val_gap
]

bars = ax.bar(experiments, train_val_gap, color=colors)
ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax.set_ylabel("Train - Val F1 Gap (percentage points)", fontsize=12)
ax.set_title("Overfitting Analysis: Train-Validation Gap by Augmentation", fontsize=14)

for bar, gap in zip(bars, train_val_gap):
    label = f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}"
    ax.annotate(
        label,
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3 if gap > 0 else -12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("outputs/figures/overfitting_analysis.png", dpi=150)
plt.savefig("outputs/figures/overfitting_analysis.pdf")
print("✓ Saved overfitting_analysis.png/pdf")

# Figure 3: Pooling Strategy Comparison (for Full Aug model)
fig, ax = plt.subplots(figsize=(12, 5))
strategies = [
    "geometric\nmean",
    "linear\nsoftmax",
    "median",
    "attention",
    "conf.\nweighted",
    "mean",
    "topk_5",
    "topk_3",
    "trimmed\nmean",
    "max",
    "majority",
]
pooling_f1 = [
    26.77,
    26.77,
    26.26,
    25.67,
    25.51,
    25.40,
    25.38,
    25.33,
    25.27,
    24.06,
    23.64,
]

colors = [
    "#2ecc71" if f > 26 else "#3498db" if f > 25 else "#95a5a6" for f in pooling_f1
]
bars = ax.barh(strategies, pooling_f1, color=colors)
ax.set_xlabel("Macro F1 Score (%)", fontsize=12)
ax.set_title(
    "Pooling Strategy Comparison (Full Augmentation Model, 3s Windows)", fontsize=14
)
ax.set_xlim(22, 28)

for bar, f1 in zip(bars, pooling_f1):
    ax.annotate(
        f"{f1:.2f}%",
        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("outputs/figures/pooling_comparison.png", dpi=150)
plt.savefig("outputs/figures/pooling_comparison.pdf")
print("✓ Saved pooling_comparison.png/pdf")

print("\n✓ All figures saved to outputs/figures/")
