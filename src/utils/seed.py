"""Random seed utilities for reproducible experiments.

Sets seeds for Python's random module, NumPy, and PyTorch to ensure
deterministic behavior across runs. Critical for comparing experiments fairly.

Example:
    >>> from src.utils.seed import set_seed
    >>> set_seed(42)
    >>> # Now all random operations will be deterministic
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Ensures reproducible results across runs when using the same seed.
    This is critical for scientific experiments and debugging.

    Args:
        seed: Random seed value (default: 42).

    Example:
        >>> set_seed(42)
        >>> torch.rand(3)
        tensor([0.8823, 0.9150, 0.3829])  # Same values every time with seed=42

    Note:
        Some operations (especially on GPU) may still have non-deterministic
        behavior. See set_deterministic() for stricter reproducibility.
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed (CPU)
    torch.manual_seed(seed)

    # Set PyTorch's random seed (GPU - both CUDA and MPS)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Note: MPS (Apple Silicon) uses the same seed as CPU
    # torch.manual_seed() affects MPS backend automatically


def set_deterministic(enabled: bool = True) -> None:
    """Enable or disable deterministic algorithms in PyTorch.

    When enabled, PyTorch will use deterministic algorithms where possible,
    which may be slower but ensures reproducibility. Some operations
    don't have deterministic implementations and will raise errors.

    Args:
        enabled: If True, enable deterministic mode. If False, disable.

    Example:
        >>> set_seed(42)
        >>> set_deterministic(True)  # Strictest reproducibility

    Warning:
        This may significantly slow down training and some operations
        may not be available in deterministic mode. Use for final
        experiments when exact reproducibility is critical.
    """
    if enabled:
        # Use deterministic algorithms
        torch.use_deterministic_algorithms(True)

        # Additional settings for CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        # Allow non-deterministic algorithms (faster)
        torch.use_deterministic_algorithms(False)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True  # Auto-tune for performance


def worker_init_fn(worker_id: int) -> None:
    """Initialize random seeds for DataLoader workers.

    When using PyTorch's DataLoader with multiple workers (num_workers > 0),
    each worker process needs its own random seed to ensure reproducibility.
    This function is passed to DataLoader as the worker_init_fn argument.

    Args:
        worker_id: Worker process ID (automatically provided by DataLoader).

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn  # Use this function
        ... )

    Note:
        Each worker gets a different seed: base_seed + worker_id
        This ensures workers produce different random augmentations.
    """
    # Get the base seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32

    # Add worker_id to make each worker's seed unique
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_rng_state() -> dict:
    """Get the current state of all random number generators.

    Useful for checkpointing - you can save and restore the exact
    RNG state to resume training with perfect reproducibility.

    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch.

    Example:
        >>> state = get_rng_state()
        >>> # ... do some random operations ...
        >>> set_rng_state(state)  # Restore to exact same state
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict) -> None:
    """Restore random number generator states from a saved state.

    Restores the exact RNG state saved by get_rng_state().

    Args:
        state: Dictionary of RNG states from get_rng_state().

    Example:
        >>> # Save state before random operations
        >>> saved_state = get_rng_state()
        >>> x = torch.rand(10)  # Random values
        >>>
        >>> # Restore state
        >>> set_rng_state(saved_state)
        >>> y = torch.rand(10)  # Same random values as x!
        >>> assert torch.allclose(x, y)
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Convenience function to set all seeds and optionally enable deterministic mode.

    Combines set_seed() and set_deterministic() into one call.

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic algorithms (slower but reproducible).

    Example:
        >>> # For experiments - fast but reproducible
        >>> seed_everything(42, deterministic=False)
        >>>
        >>> # For final results - strictest reproducibility
        >>> seed_everything(42, deterministic=True)
    """
    set_seed(seed)
    set_deterministic(deterministic)

    print(f"🌱 Random seed set to: {seed}")
    if deterministic:
        print("⚠️  Deterministic mode enabled (may be slower)")
