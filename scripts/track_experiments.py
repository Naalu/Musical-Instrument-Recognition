#!/usr/bin/env python3
"""Real-time experiment tracker for monitoring ablation study progress.

This script monitors the outputs/runs directory and provides live updates
on experiment status, progress, and preliminary results.

Usage:
    # Watch for new experiments and updates
    python scripts/track_experiments.py

    # Check status once and exit
    python scripts/track_experiments.py --no-watch

    # Custom check interval (default: 30 seconds)
    python scripts/track_experiments.py --interval 60
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def get_experiment_status(exp_dir: Path) -> Dict:
    """Get current status of an experiment.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Dictionary with experiment status information
    """
    status = {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "state": "unknown",
        "current_epoch": None,
        "total_epochs": None,
        "best_val_f1": None,
        "last_update": None,
    }

    # Check for experiment_summary.json (created when complete)
    summary_file = exp_dir / "experiment_summary.json"
    if summary_file.exists():
        status["state"] = "completed"

        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)

            status["total_epochs"] = summary.get("training", {}).get("total_epochs", 0)
            status["best_val_f1"] = summary.get("best_metrics", {}).get("val_f1", 0.0)
            status["last_update"] = datetime.fromtimestamp(
                summary_file.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")

        except Exception:
            pass

        return status

    # Check for training_history.csv (created during training)
    history_file = exp_dir / "training_history.csv"
    if history_file.exists():
        status["state"] = "running"
        status["last_update"] = datetime.fromtimestamp(
            history_file.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Try to read progress from history
        try:
            with open(history_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has header + at least one data line
                    last_line = lines[-1].strip().split(",")
                    status["current_epoch"] = int(last_line[0])
                    status["best_val_f1"] = float(last_line[3])  # val_f1 column
        except Exception:
            pass

        return status

    # Check for checkpoint files (created during training, even before history)
    checkpoint_files = list(exp_dir.glob("**/*.pth"))
    if checkpoint_files:
        status["state"] = "running"
        # Get the most recent checkpoint
        most_recent = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        status["last_update"] = datetime.fromtimestamp(
            most_recent.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Try to infer epoch from checkpoint filename
        try:
            if "epoch" in most_recent.stem:
                epoch_str = most_recent.stem.split("epoch_")[1].split("_")[0]
                status["current_epoch"] = int(epoch_str)
        except Exception:
            pass

        return status

    # Check if directory is very recent (created in last 5 minutes) - likely running
    if exp_dir.exists():
        dir_age = datetime.now().timestamp() - exp_dir.stat().st_mtime
        if dir_age < 300:  # 5 minutes
            status["state"] = "starting"
            status["last_update"] = datetime.fromtimestamp(
                exp_dir.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")

    return status


def get_experiment_type(exp_name: str) -> str:
    """Determine experiment type from directory name.

    Args:
        exp_name: Experiment directory name

    Returns:
        Human-readable experiment type
    """
    if "baseline_no_aug" in exp_name:
        return "Baseline"
    elif "pitch_only" in exp_name:
        return "Pitch Only"
    elif "stretch_only" in exp_name:
        return "Stretch Only"
    elif "specaug_only" in exp_name:
        return "SpecAug Only"
    elif "full_augment" in exp_name:
        return "Full Aug"
    else:
        return "Unknown"


def print_status_table(experiments: List[Dict]):
    """Print formatted status table for all experiments.

    Args:
        experiments: List of experiment status dictionaries
    """
    print("\n" + "=" * 100)
    print("EXPERIMENT STATUS DASHBOARD")
    print("=" * 100)

    # Header
    print(
        f"{'Experiment Type':<20} {'State':<12} {'Epoch':<10} "
        f"{'Best Val F1':<15} {'Last Update':<20}"
    )
    print("-" * 100)

    # Sort by experiment type
    exp_order = ["Baseline", "Pitch Only", "Stretch Only", "SpecAug Only", "Full Aug"]

    def sort_key(exp):
        exp_type = get_experiment_type(exp["name"])
        if exp_type in exp_order:
            return (exp_order.index(exp_type), exp["name"])
        return (999, exp["name"])

    experiments.sort(key=sort_key)

    # Print each experiment
    for exp in experiments:
        exp_type = get_experiment_type(exp["name"])
        state = exp["state"].upper()

        # Color code state (using ANSI colors)
        if exp["state"] == "completed":
            state_colored = f"\033[92m{state}\033[0m"  # Green
        elif exp["state"] == "running":
            state_colored = f"\033[93m{state}\033[0m"  # Yellow
        else:
            state_colored = f"\033[91m{state}\033[0m"  # Red

        # Format epoch info
        if exp["current_epoch"] is not None:
            if exp["total_epochs"] is not None:
                epoch_str = f"{exp['current_epoch']}/{exp['total_epochs']}"
            else:
                epoch_str = f"{exp['current_epoch']}/?"
        else:
            epoch_str = "N/A"

        # Format F1 score
        if exp["best_val_f1"] is not None:
            f1_str = f"{exp['best_val_f1'] * 100:.2f}%"
        else:
            f1_str = "N/A"

        # Format last update
        update_str = exp["last_update"] or "N/A"

        print(
            f"{exp_type:<20} {state_colored:<20} {epoch_str:<10} "
            f"{f1_str:<15} {update_str:<20}"
        )

    print("=" * 100)

    # Summary statistics
    completed = sum(1 for e in experiments if e["state"] == "completed")
    running = sum(1 for e in experiments if e["state"] == "running")
    total = len(experiments)

    print(f"\nSummary: {completed} completed, {running} running, {total} total")

    # Best result so far
    completed_exps = [e for e in experiments if e["best_val_f1"] is not None]
    if completed_exps:
        best_exp = max(completed_exps, key=lambda e: e["best_val_f1"])
        best_type = get_experiment_type(best_exp["name"])
        print(f"Best so far: {best_exp['best_val_f1'] * 100:.2f}% ({best_type})")


def check_required_experiments() -> Dict[str, bool]:
    """Check which required ablation experiments have been run.

    Returns:
        Dictionary mapping experiment types to completion status
    """
    required = {
        "Baseline": False,
        "Pitch Only": False,
        "Stretch Only": False,
        "SpecAug Only": False,
        "Full Aug": False,
    }

    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        return required

    for exp_dir in runs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.exists():
            exp_type = get_experiment_type(exp_dir.name)
            if exp_type in required:
                required[exp_type] = True

    return required


def print_next_steps():
    """Print suggested next steps based on experiment status."""
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)

    required = check_required_experiments()

    missing = [exp_type for exp_type, done in required.items() if not done]

    if not missing:
        print("✓ All required ablation experiments completed!")
        print("\nNext steps:")
        print("  1. Run results analysis: python scripts/collect_ablation_results.py")
        print("  2. Review figures in outputs/ablation_study/figures/")
        print("  3. Check LaTeX table in outputs/ablation_study/tables/")
    else:
        print(f"Missing {len(missing)} experiments:")

        commands = {
            "Baseline": "python scripts/train_augmented.py --no-augment --two-stage",
            "Pitch Only": "python scripts/train_augmented.py --pitch-only --two-stage",
            "Stretch Only": "python scripts/train_augmented.py --stretch-only --two-stage",
            "SpecAug Only": "python scripts/train_augmented.py --specaug-only --two-stage",
            "Full Aug": "python scripts/train_augmented.py --full-augment --two-stage",
        }

        for exp_type in missing:
            print(f"\n  {exp_type}:")
            print(f"    {commands[exp_type]}")


def watch_experiments(interval: int = 30):
    """Continuously monitor experiments.

    Args:
        interval: Check interval in seconds
    """
    print("\nWatching for experiment updates... (Ctrl+C to stop)")
    print(f"Check interval: {interval} seconds")

    try:
        while True:
            runs_dir = Path("outputs/runs")

            if runs_dir.exists():
                exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                experiments = [get_experiment_status(d) for d in exp_dirs]

                # Clear screen (works on Unix and Windows)
                print("\033[2J\033[H", end="")

                print_status_table(experiments)
                print_next_steps()

                print(f"\nLast checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Next check in {interval} seconds...")
            else:
                print(f"\nWaiting for experiments directory: {runs_dir}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor ablation study experiment progress"
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Check once and exit (don't watch continuously)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)",
    )

    args = parser.parse_args()

    runs_dir = Path("outputs/runs")

    if not runs_dir.exists():
        print(f"No experiments directory found: {runs_dir}")
        print("\nRun your first experiment:")
        print("  python scripts/train_augmented.py --pitch-only --two-stage")
        return

    # Get current status
    exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    experiments = [get_experiment_status(d) for d in exp_dirs]

    if not experiments:
        print("No experiments found.")
        print("\nRun your first experiment:")
        print("  python scripts/train_augmented.py --pitch-only --two-stage")
        return

    print_status_table(experiments)
    print_next_steps()

    # Watch mode
    if not args.no_watch:
        watch_experiments(args.interval)


if __name__ == "__main__":
    main()
