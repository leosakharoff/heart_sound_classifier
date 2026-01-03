# Heart Sound Classifier

A deep learning system for classifying heart sounds as **Normal** or **Abnormal** using mel spectrograms and convolutional neural networks (CNNs).

## Overview

This project implements a complete machine learning pipeline for automated heart sound classification through digital auscultation. The system processes raw phonocardiogram (PCG) recordings, applies signal processing techniques, and uses deep learning models to detect cardiac abnormalities.

### Key Features

- **Robust Signal Processing**: Bandpass filtering (25-400 Hz) to isolate heart sounds while removing noise
- **Multiple Model Architectures**: Custom CNN, ResNet18, and attention-based models
- **Data Augmentation**: Time shifting, frequency masking, and amplitude scaling for improved generalization
- **Comprehensive Evaluation**: Accuracy, ROC-AUC, confusion matrices, and training curves
- **Interpretability**: Mel spectrogram visualization for model insights

### Technical Pipeline

```
Audio Recording → Bandpass Filter → Mel Spectrogram → CNN → Classification
```

1. **Preprocessing**: Resample to 2000 Hz, apply 5th-order Butterworth bandpass filter (25-400 Hz)
2. **Feature Extraction**: 128-bin mel spectrograms capturing time-frequency representations
3. **Deep Learning**: CNN architectures with transfer learning from ImageNet
4. **Classification**: Binary classification with probability scores

## What This Does

```
Audio Recording (.wav) → Preprocessing → Mel Spectrogram → CNN → Normal/Abnormal
```

1. **Loads** heart sound recordings (PhysioNet 2016 dataset)
2. **Preprocesses** with bandpass filtering (25-400 Hz) to isolate heart sounds
3. **Extracts** mel spectrogram features
4. **Classifies** using a CNN (custom lightweight or ResNet18)
5. **Outputs** prediction with confidence score

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/leosakharoff/heart_sound_classifier.git
cd heart_sound_classifier

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

```bash
# Create data directory
mkdir -p data/physionet_2016
cd data/physionet_2016

# Download PhysioNet 2016 training data
wget https://physionet.org/files/challenge-2016/1.0.0/training-a.zip
unzip training-a.zip
```

### Training

```bash
cd src

# Quick training (limited data for testing)
python train.py --data_dir ../data/physionet_2016 --epochs 10 --max_files 200

# Full training
python train.py --data_dir ../data/physionet_2016 --epochs 20
```

### Inference

```bash
# Classify a single file
python predict.py --audio_path /path/to/heart_sound.wav --visualize

# Classify all files in a directory
python predict.py --audio_dir /path/to/sounds/ --visualize
```

## Results

| Model | Parameters | Validation Accuracy | ROC-AUC |
|-------|-----------|---------------------|---------|
| Custom CNN (Light) | 23K | 74.58% | 0.737 |
| Custom CNN (Medium) | 200K | ~75% | ~0.75 |
| ResNet18 | 11M | ~80% | ~0.80 |

**Note**: Results are based on the PhysioNet 2016 training-a subset (405 recordings, 4,738 samples). State-of-the-art on the full dataset (~3,000 recordings) achieves ~85-90% accuracy with ensemble methods and more sophisticated architectures.

## Technical Details

### Signal Processing Pipeline

```python
1. Load audio at 2000 Hz sample rate
2. Apply 5th-order Butterworth bandpass filter (25-400 Hz)
   - Removes low-frequency baseline wander
   - Removes high-frequency noise
   - Preserves heart sounds (S1, S2) and murmurs
3. Normalize amplitude to [-1, 1]
4. Extract 128-bin mel spectrogram
5. Segment into 5-second windows (50% overlap)
```

### Model Architectures

**Custom CNN (Lightweight)**:
- 4 convolutional blocks with BatchNorm and MaxPool
- Global average pooling
- 3-layer MLP classifier with dropout
- ~23K parameters
- Fast training, suitable for edge deployment

**ResNet18 (Transfer Learning)**:
- Pretrained on ImageNet
- Modified first convolution for single-channel input
- New classification head
- ~11M parameters
- Higher accuracy, longer training time

### Data Augmentation

- Time shifting (roll signal)
- Frequency masking (mask random mel bins)
- Time masking (mask random time frames)
- Amplitude scaling

### Class Imbalance Handling

The dataset has a 71% Abnormal / 29% Normal distribution. The system uses:
- Class-weighted cross-entropy loss
- Balanced sampling in data loader
- ROC-AUC as primary evaluation metric

## Project Structure

```
heart_sound_classifier/
├── data/
│   └── physionet_2016/      # PhysioNet dataset (not tracked)
├── src/
│   ├── preprocess.py        # Audio preprocessing
│   ├── dataset.py           # PyTorch dataset
│   ├── model.py             # CNN architectures
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── models/                  # Trained models (not tracked)
├── requirements.txt
└── README.md
```

## Dataset: PhysioNet 2016

The [PhysioNet/CinC 2016 Challenge](https://physionet.org/content/challenge-2016/1.0.0/) dataset:

- **3,126 recordings** from 764 subjects
- **Binary labels**: Normal (-1) or Abnormal (1)
- **Duration**: 5-120+ seconds per recording
- **Sample rate**: 2000 Hz
- **Sources**: Multiple stethoscope types and recording conditions

This project uses the training-a subset (405 recordings) as a proof-of-concept.

## Limitations & Future Work

### Current Limitations

- **Single dataset subset**: Uses only training-a (405 files) out of 3,000+ available
- **Single train/val split**: No cross-validation for robust evaluation
- **Binary classification**: Normal vs Abnormal only (no murmur type differentiation)
- **Proof-of-concept**: Not clinically validated or production-ready

### Future Improvements

- Use full PhysioNet dataset for better generalization
- Implement k-fold cross-validation
- Explore attention-based models for interpretability
- Add cardiac cycle segmentation (S1/S2 detection)
- Multi-class classification for different murmur types
- Real-time inference optimization
- External validation on independent datasets

## References

- [PhysioNet 2016 Challenge](https://physionet.org/content/challenge-2016/1.0.0/)
- [Deep Learning for Heart Sound Classification](https://www.ahajournals.org/doi/10.1161/JAHA.120.019905)
- [librosa: Audio Analysis in Python](https://librosa.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is provided as-is for research and educational purposes.

---

**Author**: Leo Sakharoff
**Date**: January 2026
