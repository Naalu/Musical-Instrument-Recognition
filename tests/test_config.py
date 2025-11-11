"""Test the config loading and validation."""

import sys
from pathlib import Path

# Add src to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config, save_config, validate_config


def test_load_baseline_config():
    """Test loading the baseline config."""
    print("Testing: load_config()")

    config_path = project_root / "configs" / "baseline.yml"
    config = load_config(config_path)

    print(f"  ✓ Loaded config from: {config_path}")
    print(f"  ✓ Experiment name: {config['experiment']['name']}")
    print(f"  ✓ Sample rate: {config['data']['sample_rate']} Hz")
    print(f"  ✓ Number of classes: {config['data']['num_classes']}")
    print(f"  ✓ Model: {config['model']['architecture']}")
    print(f"  ✓ Augmentation enabled: {config['augment']['enabled']}")

    return config


def test_validate_config(config):
    """Test config validation."""
    print("\nTesting: validate_config()")

    try:
        validate_config(config)
        print("  ✓ Config is valid!")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
        raise


def test_save_config(config):
    """Test saving config."""
    print("\nTesting: save_config()")

    output_path = project_root / "outputs" / "test_config.yml"
    save_config(config, output_path)

    print(f"  ✓ Saved config to: {output_path}")

    # Load it back to verify
    loaded = load_config(output_path)
    print("  ✓ Re-loaded successfully")

    assert loaded["experiment"]["name"] == config["experiment"]["name"]
    print("  ✓ Content matches original")


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG MODULE TESTS")
    print("=" * 60)

    # Test 1: Load config
    config = test_load_baseline_config()

    # Test 2: Validate config
    test_validate_config(config)

    # Test 3: Save config
    test_save_config(config)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
