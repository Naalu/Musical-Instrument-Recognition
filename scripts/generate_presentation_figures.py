#!/usr/bin/env python3
"""Generate figures for the final presentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Create output directory
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# =============================================================================
# DATA
# =============================================================================

# Ablation results
experiments = [
    "Baseline",
    "Pitch\nOnly",
    "Stretch\nOnly",
    "SpecAug\nOnly",
    "Full\nAugment",
]
train_f1 = [99.55, 88.70, 89.56, 60.92, 70.00]
val_f1 = [59.25, 66.43, 66.40, 62.80, 74.79]
test_f1 = [24.80, 23.98, 25.92, 23.39, 26.77]

# Pooling results (Full Aug model)
pooling_strategies = [
    "geometric_mean",
    "linear_softmax",
    "median",
    "attention",
    "conf_weighted",
    "mean",
    "topk_5",
    "topk_3",
    "trimmed_mean",
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

# Window size results
window_sizes = ["1.0s", "2.0s", "3.0s"]
window_f1 = [22.97, 25.67, 26.77]

# =============================================================================
# FIGURE 1: Augmentation Ablation - Train vs Val
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(experiments))
width = 0.35

bars1 = ax.bar(
    x - width / 2, train_f1, width, label="Training F1", color="#3498db", alpha=0.8
)
bars2 = ax.bar(
    x + width / 2, val_f1, width, label="Validation F1", color="#2ecc71", alpha=0.8
)

ax.set_ylabel("Macro F1 Score (%)")
ax.set_title("Data Augmentation Ablation: Training vs Validation Performance")
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend(loc="upper right")
ax.set_ylim(0, 110)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )
for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Add overfitting indicator
for i, (t, v) in enumerate(zip(train_f1, val_f1)):
    gap = t - v
    color = "#e74c3c" if gap > 10 else "#27ae60"
    symbol = "↓" if gap > 0 else "↑"
    ax.annotate(
        f"{symbol}{abs(gap):.0f}",
        xy=(i, max(t, v) + 8),
        ha="center",
        fontsize=9,
        color=color,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(output_dir / "ablation_train_val.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "ablation_train_val.pdf", bbox_inches="tight")
print("✓ Saved ablation_train_val.png/pdf")
plt.close()

# =============================================================================
# FIGURE 2: Overfitting Gap Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

gaps = [t - v for t, v in zip(train_f1, val_f1)]
colors = ["#e74c3c" if g > 20 else "#f39c12" if g > 0 else "#27ae60" for g in gaps]

bars = ax.bar(experiments, gaps, color=colors, edgecolor="black", linewidth=1)
ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax.set_ylabel("Train - Val F1 Gap (percentage points)")
ax.set_title("Overfitting Analysis: Train-Validation Gap by Augmentation Strategy")
ax.set_ylim(-15, 50)

# Add value labels
for bar, gap in zip(bars, gaps):
    label = f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}"
    y_pos = gap + 2 if gap > 0 else gap - 4
    ax.annotate(
        label,
        xy=(bar.get_x() + bar.get_width() / 2, y_pos),
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#e74c3c", label="Severe overfitting (>20 pts)"),
    Patch(facecolor="#f39c12", label="Moderate overfitting (0-20 pts)"),
    Patch(facecolor="#27ae60", label="Good generalization (<0 pts)"),
]
ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.savefig(output_dir / "overfitting_analysis.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "overfitting_analysis.pdf", bbox_inches="tight")
print("✓ Saved overfitting_analysis.png/pdf")
plt.close()

# =============================================================================
# FIGURE 3: Complete Ablation (Val + Test)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(experiments))
width = 0.35

bars1 = ax.bar(
    x - width / 2, val_f1, width, label="Validation F1", color="#2ecc71", alpha=0.8
)
bars2 = ax.bar(
    x + width / 2, test_f1, width, label="Test F1", color="#e74c3c", alpha=0.8
)

ax.set_ylabel("Macro F1 Score (%)")
ax.set_title("Augmentation Ablation: Validation vs Test Performance")
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend(loc="upper left")
ax.set_ylim(0, 85)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )
for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig(output_dir / "ablation_val_test.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "ablation_val_test.pdf", bbox_inches="tight")
print("✓ Saved ablation_val_test.png/pdf")
plt.close()

# =============================================================================
# FIGURE 4: Pooling Strategy Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Sort by performance
sorted_idx = np.argsort(pooling_f1)[::-1]
sorted_strategies = [pooling_strategies[i].replace("_", "\n") for i in sorted_idx]
sorted_f1 = [pooling_f1[i] for i in sorted_idx]

colors = [
    "#27ae60" if f > 26 else "#3498db" if f > 25 else "#95a5a6" for f in sorted_f1
]

bars = ax.barh(
    sorted_strategies, sorted_f1, color=colors, edgecolor="black", linewidth=0.5
)
ax.set_xlabel("Macro F1 Score (%)")
ax.set_title("Pooling Strategy Comparison\n(Full Augmentation Model, 3s Windows)")
ax.set_xlim(22, 28)

# Add value labels
for bar, f1 in zip(bars, sorted_f1):
    ax.annotate(
        f"{f1:.2f}%",
        xy=(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2),
        va="center",
        fontsize=10,
    )

# Add range annotation
ax.annotate(
    f"Range: {max(pooling_f1) - min(pooling_f1):.2f}%",
    xy=(0.95, 0.05),
    xycoords="axes fraction",
    fontsize=11,
    ha="right",
    style="italic",
)

plt.tight_layout()
plt.savefig(output_dir / "pooling_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "pooling_comparison.pdf", bbox_inches="tight")
print("✓ Saved pooling_comparison.png/pdf")
plt.close()

# =============================================================================
# FIGURE 5: Window Size Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

colors = ["#e74c3c", "#f39c12", "#27ae60"]
bars = ax.bar(window_sizes, window_f1, color=colors, edgecolor="black", linewidth=1)

ax.set_ylabel("Macro F1 Score (%)")
ax.set_xlabel("Window Size")
ax.set_title("Effect of Window Size on Test Performance\n(3s matches training)")
ax.set_ylim(20, 30)

for bar, f1 in zip(bars, window_f1):
    ax.annotate(
        f"{f1:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3),
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(output_dir / "window_size.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "window_size.pdf", bbox_inches="tight")
print("✓ Saved window_size.png/pdf")
plt.close()

# =============================================================================
# FIGURE 6: Domain Shift Visualization
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

stages = ["Training\nF1", "Validation\nF1", "Test\nF1"]
values = [70.0, 74.79, 26.77]
colors = ["#3498db", "#27ae60", "#e74c3c"]

bars = ax.bar(stages, values, color=colors, edgecolor="black", linewidth=1)

ax.set_ylabel("Macro F1 Score (%)")
ax.set_title("The Domain Shift Challenge: Performance Across Data Splits")
ax.set_ylim(0, 90)

# Add value labels
for bar, val in zip(bars, values):
    ax.annotate(
        f"{val:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2),
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

# Add arrows showing drops
ax.annotate(
    "",
    xy=(1, 74.79),
    xytext=(0, 70.0),
    arrowprops=dict(arrowstyle="->", color="green", lw=2),
)
ax.annotate("+4.8%\n(good!)", xy=(0.5, 72), ha="center", fontsize=10, color="green")

ax.annotate(
    "",
    xy=(2, 26.77),
    xytext=(1, 74.79),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
)
ax.annotate(
    "-48 pts\n(domain shift)", xy=(1.5, 50), ha="center", fontsize=10, color="red"
)

plt.tight_layout()
plt.savefig(output_dir / "domain_shift.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "domain_shift.pdf", bbox_inches="tight")
print("✓ Saved domain_shift.png/pdf")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 50)
print("✓ All figures saved to outputs/figures/")
print("=" * 50)
print("\nGenerated files:")
for f in sorted(output_dir.glob("*.png")):
    print(f"  - {f.name}")
