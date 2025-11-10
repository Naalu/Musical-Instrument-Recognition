"""Demo script showing all core utilities working together."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.core.config import load_config
from src.core.paths import get_paths_dict, get_project_root
from src.utils.device import print_device_summary, select_device
from src.utils.seed import seed_everything


def main():
    print("=" * 70)
    print("IRMAS PROJECT - UTILITIES DEMO")
    print("=" * 70)

    # 1. Set random seed for reproducibility
    print("\n1. Setting random seed...")
    seed_everything(42, deterministic=False)

    # 2. Show project paths
    print("\n2. Project paths:")
    print(f"   Root: {get_project_root()}")
    paths = get_paths_dict()
    for name, path in paths.items():
        print(f"   {name:15s}: {path}")

    # 3. Select compute device
    print("\n3. Device selection:")
    print_device_summary()
    device = select_device()

    # 4. Load configuration
    print("\n4. Loading configuration...")
    config_path = get_project_root() / "configs" / "experiment.baseline.yml"
    config = load_config(config_path)
    print(f"   Experiment: {config['experiment']['name']}")
    print(f"   Model: {config['model']['architecture']}")
    print(f"   Batch size: {config['train']['batch_size']}")
    print(f"   Learning rate: {config['train']['learning_rate']}")

    # 5. Demo computation on device
    print("\n5. Demo computation on device...")
    x = torch.rand(3, 3, device=device)
    y = torch.rand(3, 3, device=device)
    z = torch.matmul(x, y)
    print(f"   Created tensors on: {device}")
    print(f"   Matrix multiply result:\n{z.cpu().numpy()}")

    print("\n" + "=" * 70)
    print("✅ ALL UTILITIES WORKING!")
    print("=" * 70)


if __name__ == "__main__":
    main()
