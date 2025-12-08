# Save as scripts/verify_checkpoint.py
import torch

checkpoint_path = "outputs/runs/full_augment_20251201_093013/continued/best_model.pth"

print("=" * 70)
print("CHECKPOINT VERIFICATION")
print("=" * 70)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(f"\nCheckpoint keys: {checkpoint.keys()}")
print(f"\nEpoch: {checkpoint.get('epoch', 'NOT FOUND')}")
print(f"Validation F1: {checkpoint.get('val_f1', 'NOT FOUND')}")

# Check model state dict
model_state = checkpoint.get("model_state_dict", {})
print("\nModel state dict keys (first 10):")
for i, key in enumerate(list(model_state.keys())[:10]):
    print(f"  {key}: {model_state[key].shape}")

print(f"\nTotal parameters in checkpoint: {len(model_state)}")

# Check if there's history
if "history" in checkpoint:
    history = checkpoint["history"]
    print("\nTraining history available:")
    print(f"  Total epochs: {len(history.get('train_loss', []))}")
    if history.get("val_f1"):
        print(f"  Final val F1: {history['val_f1'][-1]:.4f}")
        print(f"  Best val F1: {max(history['val_f1']):.4f}")
