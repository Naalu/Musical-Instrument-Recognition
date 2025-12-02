#!/usr/bin/env python3
"""Collect and summarize ablation study results from experiments.

This script:
1. Scans outputs/runs/ for completed experiments
2. Extracts key metrics from experiment_summary.json files
3. Generates summary tables and comparison plots
4. Exports results in paper-ready formats

Usage:
    python scripts/collect_ablation_results.py
    python scripts/collect_ablation_results.py --output-dir outputs/ablation_study
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def find_experiment_dirs(runs_dir: Path) -> List[Path]:
    """Find all experiment directories with completed results.

    Args:
        runs_dir: Path to outputs/runs directory

    Returns:
        List of experiment directory paths
    """
    exp_dirs = []

    for exp_dir in runs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Check if experiment has completed (has summary file)
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.exists():
            exp_dirs.append(exp_dir)

    return sorted(exp_dirs, key=lambda p: p.name)


def load_experiment_summary(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment summary from JSON file.

    Checks for continued training results first, falls back to original.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Dictionary with experiment metadata and results
    """
    # Check for continued training results first (takes precedence)
    continued_file = exp_dir / "experiment_summary_continued.json"
    if continued_file.exists():
        print("  (Using continued training results)")
        with open(continued_file, "r") as f:
            summary = json.load(f)
        return summary

    # Fall back to original summary
    summary_file = exp_dir / "experiment_summary.json"

    with open(summary_file, "r") as f:
        summary = json.load(f)

    return summary


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from experiment summary.

    Args:
        summary: Experiment summary dictionary

    Returns:
        Dictionary with standardized metric names
    """
    # Get augmentation configuration
    augment_config = summary.get("augmentation", {})
    augment_pitch = augment_config.get("augment_pitch", False)
    augment_stretch = augment_config.get("augment_stretch", False)
    augment_specaug = augment_config.get("augment_specaug", False)

    # Determine experiment name
    if not any([augment_pitch, augment_stretch, augment_specaug]):
        exp_name = "Baseline (No Aug)"
    elif augment_pitch and not augment_stretch and not augment_specaug:
        exp_name = "Pitch Shift Only"
    elif augment_stretch and not augment_pitch and not augment_specaug:
        exp_name = "Time Stretch Only"
    elif augment_specaug and not augment_pitch and not augment_stretch:
        exp_name = "SpecAugment Only"
    elif augment_pitch and augment_stretch and augment_specaug:
        # Check if this was continued training
        if summary.get("training", {}).get("continued_from_epoch"):
            exp_name = "Full Augmentation (100ep)"
        else:
            exp_name = "Full Augmentation"
    else:
        exp_name = "Custom"

    # Extract metrics
    best_metrics = summary.get("best_metrics", {})

    return {
        "experiment": exp_name,
        "augment_pitch": augment_pitch,
        "augment_stretch": augment_stretch,
        "augment_specaug": augment_specaug,
        "val_f1": best_metrics.get("val_f1", 0.0),
        "val_loss": best_metrics.get("val_loss", 0.0),
        "best_epoch": best_metrics.get("epoch", 0),
        "total_epochs": summary.get("training", {}).get("total_epochs", 0),
        "train_f1": best_metrics.get("train_f1", 0.0),
        "seed": summary.get("training", {}).get("seed", 42),
    }


def create_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary table from experiment results.

    Args:
        results: List of experiment metric dictionaries

    Returns:
        Pandas DataFrame with formatted results
    """
    df = pd.DataFrame(results)

    # Sort by experiment type
    exp_order = [
        "Baseline (No Aug)",
        "Pitch Shift Only",
        "Time Stretch Only",
        "SpecAugment Only",
        "Full Augmentation",
        "Full Augmentation (100ep)",
    ]

    df["exp_sort"] = df["experiment"].apply(
        lambda x: exp_order.index(x) if x in exp_order else 999
    )
    df = df.sort_values("exp_sort").drop(columns=["exp_sort"])

    # Format metrics as percentages
    df["val_f1_pct"] = (df["val_f1"] * 100).round(2)
    df["train_f1_pct"] = (df["train_f1"] * 100).round(2)
    df["overfitting_gap"] = (df["train_f1_pct"] - df["val_f1_pct"]).round(2)

    return df


def plot_f1_comparison(df: pd.DataFrame, output_path: Path):
    """Create bar plot comparing F1 scores across experiments.

    Args:
        df: Results DataFrame
        output_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    x = range(len(df))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        df["train_f1_pct"],
        width,
        label="Train F1",
        alpha=0.8,
        color="steelblue",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df["val_f1_pct"],
        width,
        label="Validation F1",
        alpha=0.8,
        color="coral",
    )

    plt.xlabel("Experiment", fontsize=12, fontweight="bold")
    plt.ylabel("Macro F1-Score (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Ablation Study: Augmentation Impact on F1-Score",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, df["experiment"], rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved F1 comparison plot: {output_path}")


def plot_overfitting_analysis(df: pd.DataFrame, output_path: Path):
    """Create plot showing overfitting gap across experiments.

    Args:
        df: Results DataFrame
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    plt.bar(
        range(len(df)),
        df["overfitting_gap"],
        color="indianred",
        alpha=0.7,
        edgecolor="black",
    )

    plt.axhline(
        y=10, color="red", linestyle="--", linewidth=2, label="High Overfitting (>10%)"
    )
    plt.axhline(
        y=5,
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Moderate Overfitting (5-10%)",
    )

    plt.xlabel("Experiment", fontsize=12, fontweight="bold")
    plt.ylabel("Overfitting Gap (Train F1 - Val F1) %", fontsize=12, fontweight="bold")
    plt.title("Ablation Study: Overfitting Analysis", fontsize=14, fontweight="bold")
    plt.xticks(range(len(df)), df["experiment"], rotation=45, ha="right")
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved overfitting analysis plot: {output_path}")


def export_latex_table(df: pd.DataFrame, output_path: Path):
    """Export results as LaTeX table for paper.

    Args:
        df: Results DataFrame
        output_path: Path to save LaTeX file
    """
    # Select and format columns for paper
    paper_df = df[
        ["experiment", "val_f1_pct", "train_f1_pct", "overfitting_gap", "best_epoch"]
    ].copy()

    paper_df.columns = [
        "Experiment",
        "Val F1 (\%)",
        "Train F1 (\%)",
        "Gap (\%)",
        "Best Epoch",
    ]

    # Generate LaTeX
    latex_str = paper_df.to_latex(
        index=False,
        float_format="%.2f",
        caption="Ablation Study Results: Impact of Data Augmentation",
        label="tab:ablation_results",
        column_format="lrrrr",
        escape=False,
    )

    # Write to file
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Saved LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and summarize ablation study results"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="outputs/runs",
        help="Directory containing experiment runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation_study",
        help="Directory to save summary results",
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY RESULTS COLLECTION")
    print("=" * 70)
    print(f"Scanning: {runs_dir}")
    print()

    # Find all experiment directories
    exp_dirs = find_experiment_dirs(runs_dir)
    print(f"Found {len(exp_dirs)} completed experiments")
    print()

    if len(exp_dirs) == 0:
        print("No completed experiments found. Run training first:")
        print("  python scripts/train_augmented.py --pitch-only --two-stage")
        return

    # Extract metrics from all experiments
    results = []
    for exp_dir in exp_dirs:
        try:
            summary = load_experiment_summary(exp_dir)
            metrics = extract_metrics(summary)
            results.append(metrics)
            print(f"✓ Loaded: {exp_dir.name}")
        except Exception as e:
            print(f"✗ Failed to load {exp_dir.name}: {e}")

    print()

    if len(results) == 0:
        print("No valid results extracted. Check experiment_summary.json files.")
        return

    # Create summary table
    df = create_summary_table(results)

    # Display summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        df[
            [
                "experiment",
                "val_f1_pct",
                "train_f1_pct",
                "overfitting_gap",
                "best_epoch",
            ]
        ].to_string(index=False)
    )
    print()

    # Export results
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Create visualizations
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_f1_comparison(df, fig_dir / "f1_comparison.png")
    plot_overfitting_analysis(df, fig_dir / "overfitting_analysis.png")

    # Export LaTeX table
    table_dir = output_dir / "tables"
    table_dir.mkdir(exist_ok=True)

    export_latex_table(df, table_dir / "ablation_results.tex")

    # Final summary
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(
        f"Best validation F1: {df['val_f1_pct'].max():.2f}% "
        f"({df.loc[df['val_f1_pct'].idxmax(), 'experiment']})"
    )
    print(
        f"Lowest overfitting: {df['overfitting_gap'].min():.2f}% "
        f"({df.loc[df['overfitting_gap'].idxmin(), 'experiment']})"
    )
    print()
    print("All results saved to:", output_dir)


if __name__ == "__main__":
    main()
