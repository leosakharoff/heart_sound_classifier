#!/usr/bin/env python3
"""
Quick Demo Script
=================
Demonstrates the heart sound classifier without requiring the full dataset.
Uses synthetic heart sounds for demonstration purposes.

This is useful for:
- Testing the pipeline quickly
- Demonstrating the concept
- Verifying  code works correctly

Usage:
    python demo.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import HeartSoundPreprocessor
from model import get_model, count_parameters
from dataset import HeartSoundDataset


def generate_synthetic_heart_sound(
    duration: float = 5.0,
    sr: int = 2000,
    heart_rate: float = 70,
    add_murmur: bool = False,
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Generate a synthetic heart sound for demonstration.
    
    This creates a simple simulation of heart sounds:
    - S1 (lub): Lower frequency, longer duration
    - S2 (dub): Higher frequency, shorter duration
    - Optional murmur: Turbulent sound between S1 and S2
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        heart_rate: Heart rate in BPM
        add_murmur: Whether to add a systolic murmur
        noise_level: Amount of background noise
        
    Returns:
        Synthetic heart sound signal
    """
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    # Heart cycle duration
    cycle_duration = 60 / heart_rate  # seconds per beat
    
    # Generate heart beats
    beat_times = np.arange(0, duration, cycle_duration)
    
    for beat_time in beat_times:
        # S1 (lub) - occurs at start of systole
        s1_time = beat_time
        s1_duration = 0.15  # 150ms
        s1_freq = 50  # Hz
        
        mask = (t >= s1_time) & (t < s1_time + s1_duration)
        envelope = np.exp(-10 * (t[mask] - s1_time) / s1_duration)
        signal[mask] += 0.8 * envelope * np.sin(2 * np.pi * s1_freq * (t[mask] - s1_time))
        
        # S2 (dub) - occurs at end of systole (~0.3s after S1)
        s2_time = beat_time + 0.3
        s2_duration = 0.10  # 100ms
        s2_freq = 80  # Hz
        
        if s2_time + s2_duration < duration:
            mask = (t >= s2_time) & (t < s2_time + s2_duration)
            envelope = np.exp(-15 * (t[mask] - s2_time) / s2_duration)
            signal[mask] += 0.6 * envelope * np.sin(2 * np.pi * s2_freq * (t[mask] - s2_time))
        
        # Murmur (if abnormal) - turbulent sound during systole
        if add_murmur:
            murmur_start = s1_time + 0.1
            murmur_end = s2_time - 0.05
            
            if murmur_end < duration:
                mask = (t >= murmur_start) & (t < murmur_end)
                murmur_duration = murmur_end - murmur_start
                
                # Crescendo-decrescendo envelope
                rel_time = (t[mask] - murmur_start) / murmur_duration
                envelope = 4 * rel_time * (1 - rel_time)  # Parabolic shape
                
                # Mix of frequencies (turbulent flow)
                murmur = (
                    0.3 * np.sin(2 * np.pi * 120 * (t[mask] - murmur_start)) +
                    0.2 * np.sin(2 * np.pi * 180 * (t[mask] - murmur_start)) +
                    0.15 * np.sin(2 * np.pi * 240 * (t[mask] - murmur_start)) +
                    0.1 * np.random.randn(np.sum(mask))  # Noise component
                )
                signal[mask] += 0.4 * envelope * murmur
    
    # Add background noise
    signal += noise_level * np.random.randn(len(signal))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal


def demo():
    """Run the demonstration."""
    print("=" * 60)
    print("HEART SOUND CLASSIFIER - DEMO")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs('demo_output', exist_ok=True)
    
    # 1. Generate synthetic data
    print("1. Generating synthetic heart sounds...")
    
    sr = 2000
    normal_sounds = [generate_synthetic_heart_sound(
        duration=5.0, sr=sr, heart_rate=np.random.uniform(60, 80),
        add_murmur=False, noise_level=np.random.uniform(0.05, 0.15)
    ) for _ in range(30)]
    
    abnormal_sounds = [generate_synthetic_heart_sound(
        duration=5.0, sr=sr, heart_rate=np.random.uniform(65, 90),
        add_murmur=True, noise_level=np.random.uniform(0.08, 0.18)
    ) for _ in range(30)]
    
    print(f"  Generated {len(normal_sounds)} normal sounds")
    print(f"  Generated {len(abnormal_sounds)} abnormal sounds (with murmur)")
    
    # 2. Visualize examples
    print("\n2. Visualizing example sounds...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Normal waveform
    t = np.linspace(0, 5, len(normal_sounds[0]))
    axes[0, 0].plot(t, normal_sounds[0], linewidth=0.5, color='green')
    axes[0, 0].set_title('Normal Heart Sound - Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Abnormal waveform
    axes[0, 1].plot(t, abnormal_sounds[0], linewidth=0.5, color='red')
    axes[0, 1].set_title('Abnormal Heart Sound (Murmur) - Waveform')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Preprocess for spectrograms
    preprocessor = HeartSoundPreprocessor()
    
    normal_filtered = preprocessor.bandpass_filter(normal_sounds[0])
    normal_spec = preprocessor.extract_mel_spectrogram(normal_filtered)
    
    abnormal_filtered = preprocessor.bandpass_filter(abnormal_sounds[0])
    abnormal_spec = preprocessor.extract_mel_spectrogram(abnormal_filtered)
    
    # Normal spectrogram
    img1 = axes[1, 0].imshow(normal_spec, aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title('Normal Heart Sound - Mel Spectrogram')
    axes[1, 0].set_xlabel('Time frames')
    axes[1, 0].set_ylabel('Mel bins')
    plt.colorbar(img1, ax=axes[1, 0])
    
    # Abnormal spectrogram
    img2 = axes[1, 1].imshow(abnormal_spec, aspect='auto', origin='lower', cmap='magma')
    axes[1, 1].set_title('Abnormal Heart Sound - Mel Spectrogram')
    axes[1, 1].set_xlabel('Time frames')
    axes[1, 1].set_ylabel('Mel bins')
    plt.colorbar(img2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('demo_output/example_sounds.png', dpi=150)
    plt.close()
    print("  Saved: demo_output/example_sounds.png")
    
    # 3. Prepare data for training
    print("\n3. Preprocessing data...")
    
    all_specs = []
    all_labels = []
    
    for sound in normal_sounds:
        filtered = preprocessor.bandpass_filter(sound)
        spec = preprocessor.extract_mel_spectrogram(filtered)
        all_specs.append(spec)
        all_labels.append(0)
    
    for sound in abnormal_sounds:
        filtered = preprocessor.bandpass_filter(sound)
        spec = preprocessor.extract_mel_spectrogram(filtered)
        all_specs.append(spec)
        all_labels.append(1)
    
    X = np.array(all_specs)
    y = np.array(all_labels)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Spectrogram shape: {X_train[0].shape}")
    
    # 4. Create model
    print("\n4. Creating model...")
    
    model = get_model('cnn', num_classes=2)
    total_params, trainable_params = count_parameters(model)
    print(f"  Model: Custom CNN")
    print(f"  Parameters: {trainable_params:,}")
    
    # 5. Quick training
    print("\n5. Training (5 epochs)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = model.to(device)
    
    # Create datasets
    train_dataset = HeartSoundDataset(X_train, y_train, augment=True)
    val_dataset = HeartSoundDataset(X_val, y_val, augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        
        val_acc = correct / total
        print(f"  Epoch {epoch+1}/5 - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2%}")
    
    # 6. Test prediction
    print("\n6. Testing predictions...")
    
    model.eval()
    
    # Test on a new normal sound
    test_normal = generate_synthetic_heart_sound(add_murmur=False)
    test_filtered = preprocessor.bandpass_filter(test_normal)
    test_spec = preprocessor.extract_mel_spectrogram(test_filtered)
    test_spec_norm = (test_spec - test_spec.min()) / (test_spec.max() - test_spec.min())
    test_tensor = torch.FloatTensor(test_spec_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(test_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    
    print(f"\n  Test (Normal sound):")
    print(f"    Prediction: {'Normal' if pred == 0 else 'Abnormal'}")
    print(f"    P(Normal): {probs[0, 0].item():.2%}")
    print(f"    P(Abnormal): {probs[0, 1].item():.2%}")
    
    # Test on a new abnormal sound
    test_abnormal = generate_synthetic_heart_sound(add_murmur=True)
    test_filtered = preprocessor.bandpass_filter(test_abnormal)
    test_spec = preprocessor.extract_mel_spectrogram(test_filtered)
    test_spec_norm = (test_spec - test_spec.min()) / (test_spec.max() - test_spec.min())
    test_tensor = torch.FloatTensor(test_spec_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(test_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    
    print(f"\n  Test (Abnormal sound with murmur):")
    print(f"    Prediction: {'Normal' if pred == 0 else 'Abnormal'}")
    print(f"    P(Normal): {probs[0, 0].item():.2%}")
    print(f"    P(Abnormal): {probs[0, 1].item():.2%}")
    
    # 7. Save model
    print("\n7. Saving model...")
    
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
    }, 'models/demo_model.pth')
    print("  Saved: models/demo_model.pth")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("""
    
Key Takeaways:
1. Heart sounds can be represented as mel spectrograms
2. CNN models can learn to distinguish normal from abnormal
3. The pipeline: Audio → Filter → Spectrogram → CNN → Prediction

For real-world performance:
- Use PhysioNet 2016 dataset (real recordings)
- Train for more epochs with more data
- Use data augmentation
- Consider ResNet for better accuracy

Files created:
- demo_output/example_sounds.png (visualizations)
- models/demo_model.pth (trained model)
""")


if __name__ == "__main__":
    demo()
