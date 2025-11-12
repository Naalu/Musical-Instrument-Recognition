"""DenseNet121 architecture for musical instrument classification.

Implements transfer learning with ImageNet-pretrained DenseNet121.
Handles grayscale spectrogram input by repeating to 3 channels.

Example:
    >>> from src.models.densenet import create_densenet121
    >>>
    >>> model = create_densenet121(num_classes=11, pretrained=True)
    >>>
    >>> # Forward pass
    >>> import torch
    >>> x = torch.randn(32, 1, 128, 130)  # Batch of grayscale spectrograms
    >>> logits = model(x)
    >>> print(logits.shape)
    torch.Size([32, 11])
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    """DenseNet121 for instrument classification.

    Uses ImageNet-pretrained DenseNet121 as feature extractor.
    Converts single-channel spectrograms to 3-channel by repeating.

    Attributes:
        features: DenseNet121 feature extractor (convolutional layers).
        classifier: Classification head (fully connected layers).
        num_classes: Number of output classes.

    Example:
        >>> model = DenseNet121Classifier(num_classes=11, pretrained=True)
        >>> x = torch.randn(16, 1, 128, 130)  # Grayscale spectrograms
        >>> logits = model(x)
        >>> print(logits.shape)
        torch.Size([16, 11])
    """

    def __init__(
        self,
        num_classes: int = 11,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize DenseNet121 classifier.

        Args:
            num_classes: Number of output classes (default: 11 for IRMAS).
            pretrained: If True, use ImageNet pretrained weights.
            dropout_rate: Dropout rate before final classifier (default: 0.5).
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained DenseNet121
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            densenet = models.densenet121(weights=weights)
            print("Loaded ImageNet-pretrained DenseNet121 weights")
        else:
            densenet = models.densenet121(weights=None)
            print("Initialized DenseNet121 from scratch")

        # Extract feature extractor (all conv layers)
        # DenseNet structure: features -> classifier
        self.features = densenet.features

        # Get number of features from last layer
        # DenseNet121 has 1024 features after adaptive pooling
        num_features = densenet.classifier.in_features

        # Create new classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes),
        )

        print("DenseNet121 Classifier:")
        print(f"  Feature dim: {num_features}")
        print(f"  Output classes: {num_classes}")
        print(f"  Dropout rate: {dropout_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, height, width).
               Single-channel grayscale spectrograms.

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Convert grayscale (1 channel) to RGB (3 channels) by repeating
        # Shape: (batch, 1, H, W) -> (batch, 3, H, W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features
        features = self.features(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extractor (convolutional layers only).

        Useful for feature visualization or transfer learning.

        Returns:
            Feature extractor module.
        """
        return self.features

    def freeze_features(self):
        """Freeze all feature extractor layers.

        Use this to only train the classifier head (faster, less memory).
        Good for initial fine-tuning stage.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        print("Froze feature extractor - only classifier will be trained")

    def unfreeze_features(self):
        """Unfreeze all feature extractor layers.

        Use this for full model fine-tuning (slower, more memory).
        Good for second stage of training after classifier converges.
        """
        for param in self.features.parameters():
            param.requires_grad = True
        print("Unfroze feature extractor - full model will be trained")

    def unfreeze_last_layers(self, num_blocks: int = 2):
        """Unfreeze only the last N dense blocks.

        DenseNet121 has 4 dense blocks. Unfreezing the last 2 blocks
        (denseblock3 and denseblock4) gives fine-tuning capability
        while keeping early generic features frozen.

        Args:
            num_blocks: Number of final dense blocks to unfreeze (1-4).
                - 1: Only denseblock4 (~15% of params)
                - 2: denseblock3 + denseblock4 (~30% of params)
                - 3: denseblock2 + denseblock3 + denseblock4 (~50% of params)
                - 4: Unfreeze all (same as unfreeze_features())

        Example:
            >>> model = create_densenet121(pretrained=True)
            >>> model.freeze_features()  # Freeze all
            >>> model.unfreeze_last_layers(num_blocks=2)  # Unfreeze last 30%
        """
        # DenseNet121 structure in features module:
        # conv0, norm0, relu0, pool0,
        # denseblock1, transition1,
        # denseblock2, transition2,
        # denseblock3, transition3,
        # denseblock4, norm5

        # Freeze everything first
        for param in self.features.parameters():
            param.requires_grad = False

        # Map num_blocks to layers to unfreeze
        if num_blocks >= 4:
            # Unfreeze everything (same as unfreeze_features)
            for param in self.features.parameters():
                param.requires_grad = True
            print("Unfroze all feature extractor layers")
            return

        # Get layer names to unfreeze based on num_blocks
        layers_to_unfreeze = []

        if num_blocks >= 1:
            layers_to_unfreeze.extend(["denseblock4", "norm5"])
        if num_blocks >= 2:
            layers_to_unfreeze.extend(["denseblock3", "transition3"])
        if num_blocks >= 3:
            layers_to_unfreeze.extend(["denseblock2", "transition2"])

        # Unfreeze specified layers
        for name, module in self.features.named_children():
            if name in layers_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        percent = 100 * trainable / total

        print(f"Unfroze last {num_blocks} dense block(s): {layers_to_unfreeze}")
        print(f"Trainable: {trainable:,} / {total:,} parameters ({percent:.1f}%)")


def create_densenet121(
    num_classes: int = 11,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
) -> DenseNet121Classifier:
    """Create DenseNet121 model for instrument classification.

    Convenience function for creating the model.

    Args:
        num_classes: Number of output classes (default: 11).
        pretrained: Use ImageNet pretrained weights (default: True).
        dropout_rate: Dropout rate (default: 0.5).

    Returns:
        DenseNet121Classifier model.

    Example:
        >>> model = create_densenet121(num_classes=11, pretrained=True)
        >>> print(f"Model has {count_parameters(model):,} parameters")
    """
    model = DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.

    Example:
        >>> model = create_densenet121()
        >>> print(f"Trainable params: {count_parameters(model):,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes.

    Args:
        model: PyTorch model.

    Returns:
        Approximate model size in MB.

    Example:
        >>> model = create_densenet121()
        >>> print(f"Model size: {get_model_size_mb(model):.2f} MB")
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def print_model_summary(model: nn.Module):
    """Print a summary of the model architecture.

    Args:
        model: PyTorch model.

    Example:
        >>> model = create_densenet121()
        >>> print_model_summary(model)
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")
    print(f"Model size:           {get_model_size_mb(model):.2f} MB")

    print("\nLayer structure:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {num_params:,} parameters")

    print("=" * 70)
