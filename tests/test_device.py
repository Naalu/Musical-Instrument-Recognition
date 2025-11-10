"""Test device selection utilities."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.utils.device import (
    check_device_availability,
    get_device_info,
    move_to_device,
    print_device_summary,
    select_device,
)


def test_check_availability():
    """Test device availability checking."""
    print("Testing: check_device_availability()")

    availability = check_device_availability()

    print("  Device availability:")
    print(f"    MPS:  {availability['mps']}")
    print(f"    CUDA: {availability['cuda']}")
    print(f"    CPU:  {availability['cpu']}")

    # CPU should always be available
    assert availability["cpu"] == True, "CPU should always be available"
    print("  ✓ CPU is always available")

    # At least one should be available
    assert any(availability.values()), "At least one device should be available"
    print("  ✓ At least one compute device available")


def test_select_device_auto():
    """Test automatic device selection."""
    print("\nTesting: select_device() with auto-detection")

    device = select_device(verbose=True)

    print(f"  ✓ Selected device: {device}")
    print(f"  ✓ Device type: {device.type}")

    # Should be one of the valid types
    assert device.type in ["mps", "cuda", "cpu"], f"Unknown device type: {device.type}"
    print("  ✓ Device type is valid")

    return device


def test_select_device_cpu():
    """Test forcing CPU device."""
    print("\nTesting: select_device(preference='cpu')")

    device = select_device(preference="cpu", verbose=True)

    assert device.type == "cpu", "Should select CPU when forced"
    print("  ✓ CPU device forced successfully")


def test_get_device_info():
    """Test getting device information."""
    print("\nTesting: get_device_info()")

    device = select_device(verbose=False)
    info = get_device_info(device)

    print("  Device information:")
    for key, value in info.items():
        print(f"    {key}: {value}")

    # Check required fields
    assert "device" in info
    assert "type" in info
    print("  ✓ Device info contains required fields")


def test_move_to_device():
    """Test moving objects to device."""
    print("\nTesting: move_to_device()")

    device = select_device(verbose=False)

    # Test tensor
    tensor = torch.rand(3, 3)
    moved_tensor = move_to_device(tensor, device)
    assert moved_tensor.device.type == device.type, "Tensor should be on correct device"
    print(f"  ✓ Moved tensor to {device}")

    # Test dict of tensors
    batch = {
        "input": torch.rand(2, 3),
        "target": torch.rand(2),
    }
    moved_batch = move_to_device(batch, device)
    assert moved_batch["input"].device.type == device.type
    assert moved_batch["target"].device.type == device.type
    print(f"  ✓ Moved dictionary of tensors to {device}")

    # Test list of tensors
    tensor_list = [torch.rand(2, 2), torch.rand(3, 3)]
    moved_list = move_to_device(tensor_list, device)
    assert all(t.device.type == device.type for t in moved_list)
    print(f"  ✓ Moved list of tensors to {device}")


def test_device_computation():
    """Test that computation actually works on selected device."""
    print("\nTesting: Computation on selected device")

    device = select_device(verbose=False)

    # Create tensors on device
    a = torch.rand(100, 100, device=device)
    b = torch.rand(100, 100, device=device)

    # Perform computation
    c = torch.matmul(a, b)

    # Check result is on correct device
    assert c.device.type == device.type, "Result should be on same device"
    print(f"  ✓ Matrix multiplication works on {device}")
    print(f"  ✓ Result shape: {c.shape}")


def test_print_summary():
    """Test device summary printing."""
    print("\nTesting: print_device_summary()")

    print_device_summary()

    print("  ✓ Device summary printed successfully")


if __name__ == "__main__":
    print("=" * 70)
    print("DEVICE SELECTION TESTS")
    print("=" * 70)

    # Test 1: Check availability
    test_check_availability()

    # Test 2: Auto device selection
    device = test_select_device_auto()

    # Test 3: Force CPU
    test_select_device_cpu()

    # Test 4: Device info
    test_get_device_info()

    # Test 5: Move objects to device
    test_move_to_device()

    # Test 6: Computation on device
    test_device_computation()

    # Test 7: Print summary
    test_print_summary()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)

    if device.type == "mps":
        print("\n🚀 Your M1 Pro GPU is ready for training!")
        print("   Models will automatically use Apple Silicon acceleration.")
    elif device.type == "cuda":
        print("\n🚀 NVIDIA GPU detected and ready!")
        print("   Models will use CUDA acceleration.")
    else:
        print("\n⚠️  Using CPU - training will be slower.")
        print("   Consider using a machine with GPU for faster training.")
