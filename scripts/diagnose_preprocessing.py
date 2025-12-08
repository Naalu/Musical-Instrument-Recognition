# Save this as scripts/diagnose_preprocessing.py
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import IRMASDataset
from src.models.densenet import create_densenet121
from src.utils.device import select_device

# ============================================================================
# Load model
# ============================================================================
device = select_device()
checkpoint_path = "outputs/runs/full_augment_20251201_093013/continued/best_model.pth"

model = create_densenet121(num_classes=11, pretrained=False, dropout_rate=0.5)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("Model loaded successfully\n")

# ============================================================================
# Get a TRAINING sample (that the model should recognize well)
# ============================================================================
train_dir = Path("data/raw/IRMAS-TrainingData")
dataset = IRMASDataset(
    data_dir=train_dir,
    target_sr=22050,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
)

# Get first sample
mel_train, label_train = dataset[0]
print("Training sample:")
print(f"  Shape: {mel_train.shape}")
print(
    f"  Min: {mel_train.min():.2f}, Max: {mel_train.max():.2f}, Mean: {mel_train.mean():.2f}"
)
print(f"  Label: {label_train}\n")

# Run inference on training sample
with torch.no_grad():
    mel_batch = mel_train.unsqueeze(0).to(device)  # Add batch dim
    logits = model(mel_batch)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

print("Training sample prediction:")
print(f"  Predicted: {pred}, Ground truth: {label_train}")
print(f"  Confidence: {conf:.4f}")
print(f"  Top 3 classes: {torch.topk(probs[0], 3).indices.tolist()}")
print()

# ============================================================================
# Now test on a TEST sample (manual preprocessing to match training EXACTLY)
# ============================================================================
test_dir = Path("data/raw/IRMAS-TestingData-Part1/Part1")
test_wav = list(test_dir.glob("*.wav"))[0]
print(f"Test file: {test_wav.name}")

# Load audio EXACTLY as training does
audio, sr = librosa.load(test_wav, sr=22050, mono=True)

# Take first 3 seconds (like training)
audio_3sec = audio[: 22050 * 3]

# Extract mel-spec EXACTLY as training does
mel_spec = librosa.feature.melspectrogram(
    y=audio_3sec,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
)

# Convert to dB EXACTLY as training does
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Convert to tensor
mel_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0)

print("\nTest sample (manual preprocessing):")
print(f"  Shape: {mel_tensor.shape}")
print(
    f"  Min: {mel_tensor.min():.2f}, Max: {mel_tensor.max():.2f}, Mean: {mel_tensor.mean():.2f}"
)

# Run inference
with torch.no_grad():
    mel_batch = mel_tensor.unsqueeze(0).to(device)
    logits = model(mel_batch)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

print("\nTest sample prediction:")
print(f"  Predicted class: {pred}")
print(f"  Confidence: {conf:.4f}")
print(f"  Top 3 classes: {torch.topk(probs[0], 3).indices.tolist()}")
print(f"  Full probabilities: {probs[0].cpu().numpy()}")
