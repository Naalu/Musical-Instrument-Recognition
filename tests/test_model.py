"""Test DenseNet121 model architecture."""

import sys
from pathlib import Path

import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.densenet import (
    count_parameters,
    create_densenet121,
    get_model_size_mb,
    print_model_summary,
)
from src.utils.device import select_device


def test_model_creation():
    """Test basic model creation."""
    print("Testing: Model creation")

    # Create model without pretrained weights (faster for testing)
    model = create_densenet121(num_classes=11, pretrained=False, dropout_rate=0.5)

    print("  Model created successfully")
    print(f"  Number of classes: {model.num_classes}")

    assert model.num_classes == 11, "Should have 11 output classes"

    print("  ✓ Model creation works")


def test_forward_pass():
    """Test forward pass with dummy input."""
    print("\nTesting: Forward pass")

    model = create_densenet121(num_classes=11, pretrained=False)
    model.eval()

    # Create dummy input (batch of 4 grayscale spectrograms)
    batch_size = 4
    channels = 1  # Grayscale
    height = 128  # mel bands
    width = 130  # time frames

    x = torch.randn(batch_size, channels, height, width)

    print(f"  Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")

    # Verify output shape
    assert logits.shape == (batch_size, 11), f"Expected (4, 11), got {logits.shape}"
    assert logits.dtype == torch.float32, "Output should be float32"

    print("  ✓ Forward pass works correctly")


def test_grayscale_to_rgb_conversion():
    """Test that model handles grayscale input correctly."""
    print("\nTesting: Grayscale to RGB conversion")

    model = create_densenet121(num_classes=11, pretrained=False)
    model.eval()

    # Test with 1 channel (grayscale)
    x_gray = torch.randn(2, 1, 128, 130)
    print(f"  Input (grayscale): {x_gray.shape}")

    with torch.no_grad():
        logits_gray = model(x_gray)

    print(f"  Output: {logits_gray.shape}")

    # Test with 3 channels (should also work)
    x_rgb = torch.randn(2, 3, 128, 130)
    print(f"  Input (RGB): {x_rgb.shape}")

    with torch.no_grad():
        logits_rgb = model(x_rgb)

    print(f"  Output: {logits_rgb.shape}")

    assert logits_gray.shape == logits_rgb.shape, (
        "Both should produce same output shape"
    )

    print("  ✓ Grayscale conversion works")


def test_freeze_unfreeze():
    """Test freezing and unfreezing layers."""
    print("\nTesting: Freeze/unfreeze features")

    model = create_densenet121(num_classes=11, pretrained=False)

    # Count trainable params initially
    initial_trainable = count_parameters(model)
    print(f"  Initial trainable params: {initial_trainable:,}")

    # Freeze features
    model.freeze_features()
    frozen_trainable = count_parameters(model)
    print(f"  After freezing features: {frozen_trainable:,}")

    assert frozen_trainable < initial_trainable, (
        "Should have fewer trainable params after freezing"
    )

    # Unfreeze features
    model.unfreeze_features()
    unfrozen_trainable = count_parameters(model)
    print(f"  After unfreezing features: {unfrozen_trainable:,}")

    assert unfrozen_trainable == initial_trainable, (
        "Should restore original trainable params"
    )

    print("  ✓ Freeze/unfreeze works correctly")


def test_on_device():
    """Test model on available device (CPU or MPS)."""
    print("\nTesting: Model on device")

    device = select_device()
    print(f"  Using device: {device}")

    model = create_densenet121(num_classes=11, pretrained=False)
    model = model.to(device)
    model.eval()

    # Create input on device
    x = torch.randn(2, 1, 128, 130, device=device)
    print(f"  Input device: {x.device}")

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print(f"  Output device: {logits.device}")
    print(f"  Output shape: {logits.shape}")

    assert logits.device.type == device.type, "Output should be on same device as input"

    print("  ✓ Model works on device")


def test_parameter_count():
    """Test parameter counting utilities."""
    print("\nTesting: Parameter counting")

    model = create_densenet121(num_classes=11, pretrained=False)

    total_params = count_parameters(model)
    print(f"  Total trainable parameters: {total_params:,}")

    model_size = get_model_size_mb(model)
    print(f"  Model size: {model_size:.2f} MB")

    assert total_params > 0, "Should have trainable parameters"
    assert model_size > 0, "Should have non-zero size"

    print("  ✓ Parameter counting works")


def test_model_summary():
    """Test model summary printing."""
    print("\nTesting: Model summary")

    model = create_densenet121(num_classes=11, pretrained=False)

    print_model_summary(model)

    print("  ✓ Model summary printed")


def test_batch_sizes():
    """Test model with different batch sizes."""
    print("\nTesting: Different batch sizes")

    model = create_densenet121(num_classes=11, pretrained=False)
    model.eval()

    batch_sizes = [1, 4, 16, 32]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 128, 130)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, 11), f"Failed for batch_size={batch_size}"
        print(f"  ✓ Batch size {batch_size}: {logits.shape}")

    print("  ✓ All batch sizes work correctly")


def test_pretrained_weights():
    """Test loading pretrained weights."""
    print("\nTesting: Pretrained weights loading")

    # This will download pretrained weights (~30MB)
    print("  Downloading ImageNet pretrained weights...")
    model = create_densenet121(num_classes=11, pretrained=True)

    print("  ✓ Pretrained weights loaded successfully")

    # Test forward pass with pretrained model
    model.eval()
    x = torch.randn(2, 1, 128, 130)

    with torch.no_grad():
        logits = model(x)

    print(f"  Forward pass output: {logits.shape}")
    print("  ✓ Pretrained model works")


if __name__ == "__main__":
    print("=" * 70)
    print("DENSENET121 MODEL TESTS")
    print("=" * 70)

    # Test 1: Model creation
    test_model_creation()

    # Test 2: Forward pass
    test_forward_pass()

    # Test 3: Grayscale conversion
    test_grayscale_to_rgb_conversion()

    # Test 4: Freeze/unfreeze
    test_freeze_unfreeze()

    # Test 5: Device handling
    test_on_device()

    # Test 6: Parameter counting
    test_parameter_count()

    # Test 7: Model summary
    test_model_summary()

    # Test 8: Batch sizes
    test_batch_sizes()

    # Test 9: Pretrained weights (downloads ~30MB)
    print("\n" + "=" * 70)
    print("OPTIONAL: Test pretrained weights (downloads ~30MB)")
    response = input("Download and test pretrained weights? (y/n): ")
    if response.lower() == "y":
        test_pretrained_weights()
    else:
        print("Skipped pretrained weights test")

    print("\n" + "=" * 70)
    print("✅ ALL MODEL TESTS PASSED!")
    print("=" * 70)
    print("\n🏗️  DenseNet121 model is ready!")
    print("   - Transfer learning from ImageNet ✓")
    print("   - Grayscale spectrogram input ✓")
    print("   - 11-class instrument classification ✓")
