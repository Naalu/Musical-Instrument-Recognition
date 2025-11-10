"""Configuration loading and validation for IRMAS experiments.

This module handles YAML config files that define experiments, including
data paths, model architecture, training hyperparameters, and augmentation settings.

Example:
    >>> config = load_config('configs/experiment.baseline.yml')
    >>> print(config['experiment']['name'])
    baseline_densenet121
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If YAML parsing fails.

    Example:
        >>> cfg = load_config('configs/experiment.baseline.yml')
        >>> cfg['data']['sample_rate']
        22050
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary against expected schema.

    Checks that required keys exist and values are valid. Raises descriptive
    errors for missing or invalid configuration parameters.

    Args:
        config: Configuration dictionary from load_config().

    Raises:
        ValueError: If required keys are missing or values are invalid.
        TypeError: If value types don't match expected types.

    Example:
        >>> cfg = load_config('configs/experiment.baseline.yml')
        >>> validate_config(cfg)  # Raises ValueError if invalid
    """
    # Check top-level sections exist
    required_sections = ["experiment", "data", "features", "model", "train"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate data section
    data_cfg = config["data"]
    if "sample_rate" not in data_cfg:
        raise ValueError("Missing 'data.sample_rate' in config")
    if data_cfg["sample_rate"] <= 0:
        raise ValueError(
            f"Invalid sample_rate: {data_cfg['sample_rate']} (must be > 0)"
        )

    if "num_classes" not in data_cfg:
        raise ValueError("Missing 'data.num_classes' in config")
    if data_cfg["num_classes"] <= 0:
        raise ValueError(
            f"Invalid num_classes: {data_cfg['num_classes']} (must be > 0)"
        )

    # Validate features section
    features_cfg = config["features"]
    if "n_mels" not in features_cfg:
        raise ValueError("Missing 'features.n_mels' in config")
    if features_cfg["n_mels"] <= 0:
        raise ValueError(f"Invalid n_mels: {features_cfg['n_mels']} (must be > 0)")

    if "n_fft" not in features_cfg:
        raise ValueError("Missing 'features.n_fft' in config")
    if "hop_length" not in features_cfg:
        raise ValueError("Missing 'features.hop_length' in config")

    # Validate model section
    model_cfg = config["model"]
    if "architecture" not in model_cfg:
        raise ValueError("Missing 'model.architecture' in config")
    if "num_classes" not in model_cfg:
        raise ValueError("Missing 'model.num_classes' in config")

    # Check consistency: data.num_classes should match model.num_classes
    if data_cfg["num_classes"] != model_cfg["num_classes"]:
        raise ValueError(
            f"Mismatch: data.num_classes ({data_cfg['num_classes']}) != "
            f"model.num_classes ({model_cfg['num_classes']})"
        )

    # Validate train section
    train_cfg = config["train"]
    if "batch_size" not in train_cfg:
        raise ValueError("Missing 'train.batch_size' in config")
    if train_cfg["batch_size"] <= 0:
        raise ValueError(f"Invalid batch_size: {train_cfg['batch_size']} (must be > 0)")

    if "learning_rate" not in train_cfg:
        raise ValueError("Missing 'train.learning_rate' in config")
    if train_cfg["learning_rate"] <= 0:
        raise ValueError(
            f"Invalid learning_rate: {train_cfg['learning_rate']} (must be > 0)"
        )

    if "num_epochs" not in train_cfg:
        raise ValueError("Missing 'train.num_epochs' in config")
    if train_cfg["num_epochs"] <= 0:
        raise ValueError(f"Invalid num_epochs: {train_cfg['num_epochs']} (must be > 0)")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config recursively.

    This is useful for inheriting from a base config and only changing
    specific parameters (e.g., changing learning rate for a new experiment).

    Args:
        base: Base configuration dictionary.
        override: Override configuration (partial is OK).

    Returns:
        Merged configuration dictionary (base is not modified).

    Example:
        >>> base = {'train': {'lr': 1e-4, 'batch_size': 32}}
        >>> override = {'train': {'lr': 1e-3}}
        >>> merged = merge_configs(base, override)
        >>> merged['train']['lr']
        0.001
        >>> merged['train']['batch_size']  # Preserved from base
        32
    """
    import copy

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value (including lists - they replace, not merge)
            result[key] = value

    return result


def save_config(config: Dict[str, Any], output_path: str | Path) -> None:
    """Save configuration dictionary to YAML file.

    Useful for saving the exact config used for a training run,
    including any merged or computed parameters.

    Args:
        config: Configuration dictionary.
        output_path: Path where YAML will be saved.

    Raises:
        IOError: If file cannot be written.

    Example:
        >>> cfg = load_config('configs/experiment.baseline.yml')
        >>> save_config(cfg, 'outputs/runs/run_001/config.yml')
    """
    output_path = Path(output_path)

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
