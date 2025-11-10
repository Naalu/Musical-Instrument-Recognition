# IRMAS Dataset

## Overview

The IRMAS (Instrument Recognition in Musical Audio Signals) dataset is designed for training and evaluating musical instrument recognition systems.

**Citation:**

```
Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. (2012). 
"A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals", 
Proc. ISMIR (pp. 559-564).
```

**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)

**Download:** <https://www.upf.edu/web/mtg/irmas>

---

## Dataset Structure

### Training Data (`IRMAS-TrainingData/`)

- **Files:** 6,705 audio excerpts
- **Duration:** 3 seconds each
- **Format:** WAV, 44.1kHz, 16-bit
- **Content:** Single predominant instrument
- **Classes:** 11 instruments
- **Organization:** One folder per instrument class

```
IRMAS-TrainingData/
├── cel/   # Cello (388 files)
├── cla/   # Clarinet (505 files)
├── flu/   # Flute (451 files)
├── gac/   # Acoustic Guitar (637 files)
├── gel/   # Electric Guitar (760 files)
├── org/   # Organ (682 files)
├── pia/   # Piano (721 files)
├── sax/   # Saxophone (626 files)
├── tru/   # Trumpet (577 files)
├── vio/   # Violin (580 files)
└── voi/   # Voice (778 files)
```

**Filename format:** `[instrument]-[instrument_name]-[number].wav`  
Example: `010__[cla][cla][nod]2104__1.wav`

**Metadata in filename:**

- Artist/performer indicators (for track-level splitting)
- Recording session identifiers

### Test Data (`IRMAS-TestingData-Part1/`)

- **Files:** 2,874 audio excerpts
- **Duration:** 5-20 seconds (variable)
- **Format:** WAV, 44.1kHz, 16-bit
- **Content:** Polyphonic music (multiple instruments)
- **Labels:** Separate `.txt` files with same base name

```
IRMAS-TestingData-Part1/
├── 1.wav
├── 1.txt    # Contains: gel\tpia\n (electric guitar + piano)
├── 2.wav
├── 2.txt
└── ...
```

**Label format:** Tab-separated instrument codes, one per line.

---

## Instrument Classes (11 total)

| Code | Instrument       | Train Samples | Test Presence |
|------|------------------|---------------|---------------|
| cel  | Cello            | 388           | Common        |
| cla  | Clarinet         | 505           | Common        |
| flu  | Flute            | 451           | Common        |
| gac  | Acoustic Guitar  | 637           | Very Common   |
| gel  | Electric Guitar  | 760           | Very Common   |
| org  | Organ            | 682           | Moderate      |
| pia  | Piano            | 721           | Very Common   |
| sax  | Saxophone        | 626           | Common        |
| tru  | Trumpet          | 577           | Common        |
| vio  | Violin           | 580           | Common        |
| voi  | Voice            | 778           | Very Common   |

**Note:** Training data is imbalanced - voice (778) vs cello (388).

---

## Key Challenge

**Training:** Monophonic excerpts (single instrument)  
**Testing:** Polyphonic music (multiple instruments simultaneously)

This domain shift makes the task challenging and realistic.

---

## Data Preprocessing

See `src/data/irmas.py` for dataset indexing.  
See `src/audio/` for audio loading and normalization.  
See `src/features/` for spectrogram extraction.

---

## ⚠️ Important Notes

### Data Leakage Prevention

- **Track-level splitting:** Same recording/artist should not appear in both train and validation
- Filename contains artist identifiers - use for proper splitting
- See `src/data/splits.py` for implementation

### Not Committed to Git

The raw audio files are **NOT** included in this repository:

- Too large (~10GB total)
- License restrictions
- Each user must download separately

### Cache Directory

Preprocessed features (spectrograms) are cached in `data/cache/`:

- `.npy` files containing mel-spectrograms
- Speeds up training (no need to recompute features)
- Can be regenerated from raw audio if deleted

---

## Download Instructions

1. Visit: <https://www.upf.edu/web/mtg/irmas>
2. Download `IRMAS-TrainingData.zip` (~8GB)
3. Download `IRMAS-TestingData-Part1.zip` (~2GB)
4. Extract both to `data/raw/`:

```bash
   cd data/raw/
   unzip IRMAS-TrainingData.zip
   unzip IRMAS-TestingData-Part1.zip
```

Expected structure after extraction:

```
data/
├── raw/
│   ├── IRMAS-TrainingData/
│   │   ├── cel/
│   │   ├── cla/
│   │   └── ...
│   └── IRMAS-TestingData-Part1/
│       ├── 1.wav
│       ├── 1.txt
│       └── ...
└── cache/
    └── (generated during training)
```

---

## Verification

After downloading, verify dataset integrity:

```bash
python scripts/verify_dataset.py
```

This will check:

- All expected directories exist
- File counts match expected values
- Audio files are readable
- No corrupted files
