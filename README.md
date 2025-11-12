# Musical Instrument Classification with Deep Learning

Deep learning system for classifying musical instruments from audio using DenseNet121 and mel-spectrograms on the IRMAS dataset.

## Project Overview

This project implements a **production-ready deep learning pipeline** for musical instrument classification, achieving **59.25% macro F1-score** on the IRMAS dataset using transfer learning with DenseNet121.

### Key Results

- **Model**: DenseNet121 (ImageNet pretrained)
- **Dataset**: IRMAS (6,705 training + 2,874 test samples)
- **Performance**: 59.25% macro F1-score (target: 60%)
- **Training**: Two-stage approach with selective layer unfreezing
- **Hardware**: Apple M1 Pro with MPS acceleration

## Architecture

- **Input**: 3-second audio clips → Log-mel spectrograms (128 bands × 130 frames)
- **Model**: DenseNet121 with custom classifier head
- **Training Strategy**:
  1. Stage 1: Train classifier head only (5 epochs)
  2. Stage 2: Fine-tune last 30% of model (45 epochs)
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- **Loss**: Weighted CrossEntropyLoss (handles class imbalance)

## Dataset

**IRMAS (Instrument Recognition in Musical Audio Signals)**

- 11 instrument classes: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, voice
- Training: 6,705 monophonic 3-second clips
- Testing: 2,874 polyphonic excerpts with multi-label annotations
- Sample rate: 44.1 kHz → 22.05 kHz
- Audio format: WAV, 16-bit

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Musical-Instrument-Recognition.git
cd Musical-Instrument-Recognition

# Create conda environment
conda create -n irmas python=3.11
conda activate irmas

# Install dependencies
pip install torch torchvision torchaudio
pip install librosa pandas pyyaml tqdm scikit-learn matplotlib

# Download IRMAS dataset
# Visit: https://www.upf.edu/web/mtg/irmas
# Download and extract to data/raw/
```

### Training

```bash
# Two-stage training (recommended)
python scripts/train_two_stage.py

# Single-stage training
python scripts/train.py --config configs/baseline.yml
```

### Inference

```bash
# Classify a single audio file
python scripts/inference.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pth
```

## 📁 Project Structure

```
Musical-Instrument-Recognition/
├── configs/                  # Configuration files
│   └── baseline.yml
├── data/                     # Data directory
│   └── raw/                  # Raw IRMAS dataset (downloaded separately)
├── docs/                     # Documentation
├── src/                      # Source code
│   ├── audio/                # Audio loading (librosa)
│   ├── augment/              # Data augmentation (SpecAugment)
│   ├── core/                 # Config, paths
│   ├── data/                 # Dataset indexing and PyTorch Dataset
│   ├── features/             # Mel-spectrogram extraction
│   ├── metrics/              # Classification metrics
│   ├── models/               # DenseNet121 architecture
│   ├── train/                # Training loop
│   └── utils/                # Utility functions: device, seed
├── scripts/                  # Training and inference scripts
├── tests/                    # Unit tests
└── checkpoints/              # Saved model checkpoints
```

## Implementation Details

### Feature Extraction

- **Mel-spectrogram**: 128 mel bands, 2048 FFT, 512 hop length
- **Frequency range**: 0-11,025 Hz (Nyquist frequency)
- **Log-scale**: Power to dB conversion for better dynamic range
- **Grayscale**: Single-channel input (repeated to 3 channels for DenseNet)

### Model Architecture

- **Base**: DenseNet121 (7M parameters)
- **Modifications**:
  - Custom classifier: 1024 → 11 classes
  - Dropout: 0.5
  - Selective unfreezing: Last 2 dense blocks (~30% of params)
- **Input shape**: (batch, 1, 128, 130)
- **Output**: (batch, 11) logits

### Training Strategy

- **Stage 1 (5 epochs)**:
  - Freeze DenseNet features
  - Train classifier head only (11K params)
  - Learning rate: 1e-4
- **Stage 2 (45 epochs)**:
  - Unfreeze last 2 dense blocks (2M params)
  - Fine-tune with lower LR: 1e-5
  - Early stopping: patience 10 epochs

### Preventing Data Leakage

- **Artist-based splitting**: Same artist never in both train/val
- **Stratified split**: Maintains class proportions
- **Validation**: 15% of training data (860 samples)

## Results

### Baseline Performance

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 99.4% | 62.2% |
| Macro F1 | 99.4% | **59.25%** |
| Precision | 99.4% | 60.1% |
| Recall | 99.4% | 58.9% |

### Per-Class F1 Scores (Validation)

| Instrument | F1-Score | Training Samples |
|------------|----------|------------------|
| Voice (voi) | 68.2% | 661 |
| Electric Guitar (gel) | 65.8% | 647 |
| Piano (pia) | 62.4% | 612 |
| Acoustic Guitar (gac) | 61.1% | 543 |
| Saxophone (sax) | 59.7% | 533 |
| Organ (org) | 56.3% | 661 |
| Trumpet (tru) | 54.1% | 544 |
| Violin (vio) | 53.8% | 493 |
| Flute (flu) | 52.6% | 392 |
| Clarinet (cla) | 51.4% | 428 |
| Cello (cel) | 49.2% | 331 |

## Technical Challenges Solved

1. **Overfitting**: Selective unfreezing to reduce trainable params
2. **Class imbalance**: Weighted loss function (inverse frequency)
3. **Data leakage**: Artist-based train/val split
4. **Transfer learning**: Two-stage training preserves ImageNet features
5. **MPS acceleration**: Both CUDA and Native Apple Silicon Metal Performance Shader support

## Future Improvements

To reach 60%+ F1-score:

- [ ] **Data augmentation**: SpecAugment, time/frequency masking
- [ ] **Ensemble methods**: Multiple models with voting
- [ ] **Better architecture**: Try EfficientNet, ResNet variants
- [ ] **Longer training**: More epochs with early stopping

## References

1. **IRMAS Dataset**: Bosch et al. (2012) "A Dataset for Instrument Recognition in Polyphonic Music"
2. **DenseNet**: Huang et al. (2017) "Densely Connected Convolutional Networks"
3. **Transfer Learning**: Yosinski et al. (2014) "How transferable are features in deep neural networks?"
4. **Audio Classification**: Hershey et al. (2017) "CNN Architectures for Large-Scale Audio Classification"

## License

MIT License - See LICENSE file for details

## Acknowledgments

- IRMAS dataset creators at Music Technology Group, Universitat Pompeu Fabra
- PyTorch and torchvision teams
- librosa audio processing library

## Authors

**Iris Robedeaux**

- GitHub: [@iris-robedeaux](https://github.com/iris-robedeaux)

**Karl Reger**

- GitHub: [@naalu](https://github.com/naalu)

As part of our coursework for:

- CS 599 - *Deep Learning*
- Institution: *Northern Arizona University*
