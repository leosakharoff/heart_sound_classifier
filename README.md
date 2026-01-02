# Heart Sound Classifier

A deep learning prototype for classifying heart sounds as **Normal** or **Abnormal** using mel spectrograms and convolutional neural networks (CNNs).

## Project Description

This project implements an end-to-end machine learning pipeline for automated heart sound classification, designed to assist in the early detection of cardiac abnormalities through digital auscultation. The system processes raw audio recordings from stethoscopes, applies advanced signal processing techniques, and uses deep learning models to classify heart sounds with high accuracy.

### Key Features

- **Robust Signal Processing**: Implements bandpass filtering (25-400 Hz) to isolate heart sounds while removing noise and baseline wander
- **Multiple Model Architectures**: Supports custom CNN, ResNet18, and attention-based models for different performance/complexity trade-offs
- **Data Augmentation**: Enhances model generalization through time shifting, frequency masking, and amplitude scaling
- **Comprehensive Evaluation**: Provides detailed metrics including accuracy, ROC-AUC, confusion matrices, and training curves
- **Visualization**: Generates mel spectrograms and prediction visualizations for interpretability

### Technical Approach

The system follows a modern audio classification pipeline:

1. **Audio Preprocessing**: Raw audio signals are resampled to 2000 Hz and filtered using a 5th-order Butterworth bandpass filter to preserve the frequency range of heart sounds (S1, S2 heartbeats and murmurs)
2. **Feature Extraction**: Mel spectrograms are computed with 128 frequency bins, capturing time-frequency representations that are effective for audio classification
3. **Deep Learning**: CNN architectures learn hierarchical patterns from spectrograms, with transfer learning from ImageNet for improved performance
4. **Classification**: Binary classification outputs probability scores for normal vs abnormal heart sounds

### Use Cases

- **Medical Screening**: Assist healthcare providers in preliminary cardiac screening
- **Telemedicine**: Enable remote heart sound analysis for patients in underserved areas
- **Education**: Train medical students on heart sound identification
- **Research**: Provide a baseline for developing more advanced cardiac diagnostic tools

Built as a proof-of-concept for understanding digital auscultation and AI-powered diagnostic tools.

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download PhysioNet 2016 Data

```bash
# Create data directory
mkdir -p data/physionet_2016

# Download training sets (you need at least one)
cd data/physionet_2016

# Training Set A (~25 MB)
wget https://physionet.org/files/challenge-2016/1.0.0/training-a.zip
unzip training-a.zip

# Optional: Download more sets for better results
# wget https://physionet.org/files/challenge-2016/1.0.0/training-b.zip
# wget https://physionet.org/files/challenge-2016/1.0.0/training-c.zip
```

### 3. Train the Model

```bash
cd src

# Quick training (limited data, ~5 min on CPU)
python train.py --data_dir ../data/physionet_2016 --epochs 10 --max_files 200

# Full training (~15-30 min on CPU)
python train.py --data_dir ../data/physionet_2016 --epochs 20
```

### 4. Make Predictions

```bash
# Classify a single file
python predict.py --audio_path /path/to/heart_sound.wav --visualize

# Classify all files in a directory
python predict.py --audio_dir /path/to/sounds/ --visualize
```

## Expected Results

| Model | Accuracy | Training Time (CPU) |
|-------|----------|---------------------|
| Custom CNN | 75-80% | 5-10 min |
| ResNet18 | 80-85% | 10-20 min |
| CNN + Attention | 78-83% | 8-15 min |

State-of-the-art on this dataset is ~90%, achieved with:
- Larger datasets
- More sophisticated preprocessing
- Ensemble methods
- Transformer architectures

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

**Custom CNN** (lightweight, fast):
- 4 conv blocks with BatchNorm and MaxPool
- Global average pooling
- 3-layer MLP classifier with dropout
- ~500K parameters

**ResNet18** (transfer learning):
- Pretrained on ImageNet
- Modified first conv for single-channel input
- New classification head
- ~11M parameters

### Data Augmentation

During training:
- Time shifting (roll signal)
- Frequency masking (mask random mel bins)
- Time masking (mask random time frames)
- Amplitude scaling

## Project Structure

```
heart_sound_classifier/
├── data/
│   └── physionet_2016/      # PhysioNet dataset
├── src/
│   ├── preprocess.py        # Audio preprocessing
│   ├── dataset.py           # PyTorch dataset
│   ├── model.py             # CNN architectures
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── models/
│   └── best_model.pth       # Trained weights
├── predictions/             # Visualizations
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

Abnormalities include murmurs from various conditions.

## Relevance to AUSCORA

This prototype demonstrates understanding of:

1. **The technical problem**: Classifying heart sounds is challenging due to noise, variability, and subtle differences between normal/abnormal
2. **Signal processing**: Bandpass filtering, spectrogram extraction, segmentation
3. **Deep learning for audio**: CNNs on spectrograms, transfer learning
4. **Clinical context**: The goal is to assist clinicians, not replace them

### What production systems add:
- Real-time inference
- Multi-device compatibility (device-agnostic models)
- S1/S2 segmentation and murmur localization
- Integration with clinical workflows
- EU MDR regulatory compliance
- Extensive clinical validation

## References

- [PhysioNet 2016 Challenge](https://physionet.org/content/challenge-2016/1.0.0/)
- [Deep Learning for Heart Sound Classification](https://www.ahajournals.org/doi/10.1161/JAHA.120.019905)
- [librosa: Audio Analysis in Python](https://librosa.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

Built by Leo Sakharoff as part of interview preparation for AUSCORA.
January 2026
