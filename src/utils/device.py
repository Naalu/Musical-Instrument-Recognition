"""Device selection utilities for PyTorch.

Automatically detects and selects the best available compute device:
- MPS (Apple Silicon GPU) for M1/M2/M3 Macs
- CUDA (NVIDIA GPU) for systems with NVIDIA GPUs
- CPU as fallback

Example:
    >>> from src.utils.device import select_device
    >>> device = select_device()
    >>> print(device)
    mps  # On M1 Pro
    >>> model = model.to(device)
"""

from typing import Literal, Optional

import torch


def select_device(
    preference: Optional[Literal["mps", "cuda", "cpu"]] = None, verbose: bool = True
) -> torch.device:
    """Select the best available compute device.

    Tries devices in order of preference (if not specified):
    1. MPS (Apple Silicon GPU)
    2. CUDA (NVIDIA GPU)
    3. CPU (fallback)

    Args:
        preference: Force a specific device ('mps', 'cuda', or 'cpu').
                   If None, auto-detect best available device.
        verbose: If True, print device selection info.

    Returns:
        PyTorch device object ready to use with .to(device)

    Example:
        >>> # Auto-detect best device
        >>> device = select_device()
        Using device: mps (Apple M1 Pro)

        >>> # Force CPU (for debugging)
        >>> device = select_device(preference='cpu')
        Using device: cpu (forced)

    Raises:
        ValueError: If preference is specified but not available.
    """
    # If user specified a preference, try to honor it
    if preference is not None:
        preference = preference.lower()

        if preference == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError(
                    "MPS requested but not available. Check PyTorch version and macOS version."
                )
            device = torch.device("mps")
            if verbose:
                print("Using device: mps (forced)")

        elif preference == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA requested but not available. Check GPU and CUDA installation."
                )
            device = torch.device("cuda")
            if verbose:
                cuda_name = torch.cuda.get_device_name(0)
                print(f"Using device: cuda ({cuda_name}, forced)")

        elif preference == "cpu":
            device = torch.device("cpu")
            if verbose:
                print("Using device: cpu (forced)")

        else:
            raise ValueError(
                f"Unknown device preference: {preference}. Use 'mps', 'cuda', or 'cpu'."
            )

    else:
        # Auto-detect: MPS > CUDA > CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            if verbose:
                # Try to get more info about the chip
                import platform

                chip = platform.processor() or "Apple Silicon"
                print(f"Using device: mps ({chip})")

        elif torch.cuda.is_available():
            device = torch.device("cuda")
            if verbose:
                cuda_name = torch.cuda.get_device_name(0)
                cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"Using device: cuda ({cuda_name}, {cuda_memory:.1f}GB)")

        else:
            device = torch.device("cpu")
            if verbose:
                print("Using device: cpu (no GPU available)")

    return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """Get detailed information about a compute device.

    Args:
        device: Device to query. If None, queries currently selected device.

    Returns:
        Dictionary with device information including name, type, and memory.

    Example:
        >>> device = select_device()
        >>> info = get_device_info(device)
        >>> print(info['name'])
        mps
        >>> print(info['type'])
        Apple Silicon GPU
    """
    if device is None:
        device = select_device(verbose=False)

    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "mps":
        import platform

        info["name"] = "mps"
        info["description"] = "Apple Silicon GPU"
        info["chip"] = platform.processor() or "Apple Silicon"
        # MPS doesn't expose memory info easily
        info["memory_gb"] = "shared"

    elif device.type == "cuda":
        info["name"] = f"cuda:{device.index if device.index else 0}"
        info["description"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["memory_gb"] = props.total_memory / 1e9
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["multi_processor_count"] = props.multi_processor_count

    elif device.type == "cpu":
        import platform

        import psutil

        info["name"] = "cpu"
        info["description"] = "CPU"
        info["processor"] = platform.processor()
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        info["memory_gb"] = psutil.virtual_memory().total / 1e9

    return info


def check_device_availability() -> dict:
    """Check availability of all device types.

    Useful for debugging or system reports.

    Returns:
        Dictionary showing which devices are available.

    Example:
        >>> availability = check_device_availability()
        >>> print(availability)
        {'mps': True, 'cuda': False, 'cpu': True}
    """
    return {
        "mps": torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,  # CPU always available
    }


def move_to_device(obj, device: torch.device):
    """Move tensors, models, or collections to specified device.

    Handles various PyTorch objects including tensors, models, and
    nested structures (lists, tuples, dicts).

    Args:
        obj: Object to move (tensor, model, list, tuple, or dict).
        device: Target device.

    Returns:
        Object moved to device (same type as input).

    Example:
        >>> device = select_device()
        >>> model = model.to(device)
        >>>
        >>> # Move batch of tensors
        >>> batch = {'images': images, 'labels': labels}
        >>> batch = move_to_device(batch, device)
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)

    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        moved = [move_to_device(item, device) for item in obj]
        return type(obj)(moved)  # Preserve list vs tuple

    else:
        # Can't move this type, return as-is
        return obj


def print_device_summary():
    """Print a summary of all available devices.

    Useful for verifying setup and debugging device issues.

    Example:
        >>> print_device_summary()

        Device Availability:
        ✓ MPS (Apple Silicon): Available
        ✗ CUDA (NVIDIA GPU): Not available
        ✓ CPU: Available

        Selected Device: mps (Apple M1 Pro)
    """
    print("\n" + "=" * 60)
    print("DEVICE AVAILABILITY")
    print("=" * 60)

    availability = check_device_availability()

    # MPS
    if availability["mps"]:
        import platform

        chip = platform.processor() or "Apple Silicon"
        print(f"✓ MPS (Apple Silicon): Available ({chip})")
    else:
        print("✗ MPS (Apple Silicon): Not available")

    # CUDA
    if availability["cuda"]:
        cuda_name = torch.cuda.get_device_name(0)
        cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA (NVIDIA GPU): Available ({cuda_name}, {cuda_memory:.1f}GB)")
    else:
        print("✗ CUDA (NVIDIA GPU): Not available")

    # CPU (always available)
    import platform

    import psutil

    cores = psutil.cpu_count(logical=False)
    print(f"✓ CPU: Available ({cores} physical cores)")

    print("\n" + "-" * 60)

    # Show selected device
    device = select_device(verbose=False)
    info = get_device_info(device)
    print(f"Selected Device: {info['device']} ({info.get('description', 'N/A')})")
    print("=" * 60 + "\n")
