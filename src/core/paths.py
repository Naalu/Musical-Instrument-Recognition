"""Path management utilities for IRMAS project.

Provides centralized path handling for data, outputs, cache, and other directories.
All paths are relative to the project root for portability across systems.

Example:
    >>> from src.core.paths import get_data_dir, get_output_dir
    >>> data_path = get_data_dir('raw')
    >>> print(data_path)
    /path/to/Musical-Instrument-Recognition/data/raw
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get absolute path to project root directory.

    The project root is identified by finding the directory containing
    the .git folder (assuming we're in a git repository).

    Returns:
        Path object pointing to project root.

    Example:
        >>> root = get_project_root()
        >>> print(root.name)
        Musical-Instrument-Recognition
    """
    # Start from this file's location and search upward for .git
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent

    # Fallback: if not in git repo, go up 3 levels from this file
    # (paths.py is in src/core/, so ../../ gets us to project root)
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """Get path to data directory or subdirectory.

    Args:
        subdir: Optional subdirectory name (e.g., "raw", "cache").

    Returns:
        Absolute path to data directory or subdirectory.

    Example:
        >>> get_data_dir()
        PosixPath('/path/to/project/data')
        >>> get_data_dir('raw')
        PosixPath('/path/to/project/data/raw')
    """
    data_dir = get_project_root() / "data"

    if subdir:
        data_dir = data_dir / subdir

    return data_dir


def get_output_dir(run_name: Optional[str] = None, create: bool = True) -> Path:
    """Get path to outputs directory for a specific run.

    If run_name is provided, creates a timestamped directory like:
    outputs/runs/2024-11-09_19-30-45_baseline_densenet121/

    Args:
        run_name: Optional run identifier (e.g., "baseline_seed42").
        create: If True, create directory if it doesn't exist.

    Returns:
        Path to outputs/runs/<timestamped_run_name> or outputs/runs/ if run_name is None.

    Example:
        >>> get_output_dir('baseline')
        PosixPath('/path/to/project/outputs/runs/2024-11-09_19-30-45_baseline')
    """
    outputs_root = get_project_root() / "outputs" / "runs"

    if run_name is None:
        if create:
            outputs_root.mkdir(parents=True, exist_ok=True)
        return outputs_root

    # Create timestamped directory name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = outputs_root / f"{timestamp}_{run_name}"

    if create:
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def get_cache_dir(create: bool = True) -> Path:
    """Get path to feature cache directory.

    Args:
        create: If True, create directory if it doesn't exist.

    Returns:
        Path to data/cache/ directory.

    Example:
        >>> cache = get_cache_dir()
        >>> print(cache)
        /path/to/project/data/cache
    """
    cache_dir = get_data_dir("cache")

    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_figures_dir(create: bool = True) -> Path:
    """Get path to figures output directory.

    Args:
        create: If True, create directory if it doesn't exist.

    Returns:
        Path to outputs/figures/ directory.

    Example:
        >>> figs = get_figures_dir()
        >>> confusion_matrix_path = figs / 'confusion_matrix.png'
    """
    figures_dir = get_project_root() / "outputs" / "figures"

    if create:
        figures_dir.mkdir(parents=True, exist_ok=True)

    return figures_dir


def ensure_dir_exists(path: Path) -> Path:
    """Create directory if it doesn't exist (including parents).

    Args:
        path: Directory path to create.

    Returns:
        The same path object (for chaining).

    Raises:
        PermissionError: If directory cannot be created due to permissions.

    Example:
        >>> from pathlib import Path
        >>> my_dir = ensure_dir_exists(Path('outputs/experiments/exp_001'))
        >>> my_dir.exists()
        True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Convenience function to get all important paths at once
def get_paths_dict() -> dict:
    """Get dictionary of all important project paths.

    Useful for debugging or logging path configuration.

    Returns:
        Dictionary mapping path names to Path objects.

    Example:
        >>> paths = get_paths_dict()
        >>> for name, path in paths.items():
        ...     print(f"{name}: {path}")
    """
    return {
        "project_root": get_project_root(),
        "data": get_data_dir(),
        "data_raw": get_data_dir("raw"),
        "data_cache": get_cache_dir(create=False),
        "outputs": get_output_dir(create=False),
        "figures": get_figures_dir(create=False),
    }
