"""Test that all required packages import correctly."""

print("Testing environment setup...\n")

# Test PyTorch and MPS
import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"  - MPS available: {torch.backends.mps.is_available()}")
print(f"  - MPS built: {torch.backends.mps.is_built()}")

# Test torchvision (has DenseNet121)
import torchvision
print(f"✓ Torchvision {torchvision.__version__}")

# Test audio processing
import librosa
print(f"✓ librosa {librosa.__version__}")

import soundfile
print(f"✓ soundfile {soundfile.__version__}")

# Test scientific computing
import numpy as np
print(f"✓ numpy {np.__version__}")

import pandas as pd
print(f"✓ pandas {pd.__version__}")

import scipy
print(f"✓ scipy {scipy.__version__}")

# Test ML utilities
import sklearn
print(f"✓ scikit-learn {sklearn.__version__}")

# Test visualization
import matplotlib
print(f"✓ matplotlib {matplotlib.__version__}")

import seaborn
print(f"✓ seaborn {seaborn.__version__}")

# Test config handling
import yaml
print(f"✓ PyYAML {yaml.__version__}")

# Test utilities
import tqdm
print(f"✓ tqdm {tqdm.__version__}")

print("\n🎉 All packages imported successfully!")
print("\nYour environment is ready for deep learning! 🚀")
