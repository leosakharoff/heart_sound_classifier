# Heart Sound Classifier

A weekend project to understand how AI-powered stethoscopes work. Takes heart recordings, turns them into spectrograms, and classifies them as normal or abnormal using a CNN.

Proof-of-concept, not a medical device — but it works.

## Quick start

```bash
pip install -r requirements.txt

# Get data (PhysioNet 2016, ~25MB)
mkdir -p data/physionet_2016 && cd data/physionet_2016
wget https://physionet.org/files/challenge-2016/1.0.0/training-a.zip
unzip training-a.zip && cd ../..

# Train (~10 min on CPU)
python src/train.py --data_dir ./data/physionet_2016 --epochs 15

# Classify a file
python src/predict.py --audio_path ./data/physionet_2016/training-a/a0001.wav --visualize
```

## How it works

Audio → bandpass filter (25-400 Hz) → mel spectrogram → CNN → prediction

The model is small (\~23K parameters) and gets about 75% accuracy on PhysioNet 2016 data. State-of-the-art is 85-90% with bigger models and more data.

## What I learned

* Real data matters — synthetic heart sounds didn't work at all
* Smaller models generalize better on limited data (big models overfit immediately)
* Class imbalance needs careful handling (71% of samples are "abnormal")

## Limitations

This is a proof-of-concept: single dataset subset, no cross-validation, binary classification only. Production systems need way more validation and regulatory approval.

---

Leo Sakharoff, January 2026
