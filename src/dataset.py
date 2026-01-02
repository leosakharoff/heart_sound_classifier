"""
Heart Sound Dataset Module
==========================
PyTorch Dataset and DataLoader implementations for heart sound classification.

This module provides efficient data loading and augmentation for training
deep learning models on heart sound spectrograms.

Features:
- Efficient memory management for large datasets
- On-the-fly data augmentation during training
- Support for train/validation/test splits
- Configurable batch sizes and shuffling

Data Augmentation Techniques:
1. Time Shifting: Roll the spectrogram along time axis
   - Simulates different recording positions
   - Helps model learn temporal invariance

2. Frequency Masking: Mask random frequency bands
   - Forces model to use multiple frequency features
   - Improves robustness to frequency-specific noise

3. Time Masking: Mask random time frames
   - Encourages model to use temporal context
   - Prevents overfitting to specific patterns

4. Amplitude Scaling: Randomly scale spectrogram values
   - Simulates different recording volumes
   - Improves generalization

Usage:
    from dataset import HeartSoundDataset, create_data_loaders

    # Create dataset
    dataset = HeartSoundDataset(spectrograms, labels, augment=True)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable
import random


class HeartSoundDataset(Dataset):
    """
    PyTorch Dataset for heart sound spectrograms.
    
    Handles:
    - Loading preprocessed spectrograms
    - Data augmentation (time shift, frequency masking)
    - Converting to PyTorch tensors
    """
    
    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            spectrograms: Array of mel spectrograms (N, n_mels, time_frames)
            labels: Array of labels (0=normal, 1=abnormal)
            transform: Optional transform function
            augment: Whether to apply data augmentation
        """
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (spectrogram tensor, label)
        """
        spec = self.spectrograms[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment:
            spec = self._augment(spec)
        
        # Normalize spectrogram to [0, 1] range
        spec = self._normalize(spec)
        
        # Convert to tensor with channel dimension
        # Shape: (1, n_mels, time_frames) - single channel image
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        
        # Apply optional transform
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        return spec_tensor, label
    
    def _normalize(self, spec: np.ndarray) -> np.ndarray:
        """Normalize spectrogram to [0, 1] range."""
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min > 0:
            return (spec - spec_min) / (spec_max - spec_min)
        return spec - spec_min
    
    def _augment(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to spectrogram.
        
        Augmentations help the model generalize better:
        - Time shift: Simulate recordings starting at different points
        - Frequency masking: Make model robust to missing frequency bands
        - Time masking: Make model robust to brief interruptions
        """
        # Time shift (roll along time axis)
        if random.random() > 0.5:
            shift = random.randint(-spec.shape[1]//10, spec.shape[1]//10)
            spec = np.roll(spec, shift, axis=1)
        
        # Frequency masking (mask random frequency bands)
        if random.random() > 0.5:
            num_mels = spec.shape[0]
            mask_width = random.randint(1, num_mels // 8)
            mask_start = random.randint(0, num_mels - mask_width)
            spec[mask_start:mask_start + mask_width, :] = spec.min()
        
        # Time masking (mask random time segments)
        if random.random() > 0.5:
            num_frames = spec.shape[1]
            mask_width = random.randint(1, num_frames // 8)
            mask_start = random.randint(0, num_frames - mask_width)
            spec[:, mask_start:mask_start + mask_width] = spec.min()
        
        # Random amplitude scaling
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            spec = spec * scale
        
        return spec


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        X_train: Training spectrograms
        y_train: Training labels
        X_val: Validation spectrograms
        y_val: Validation labels
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = HeartSoundDataset(X_train, y_train, augment=True)
    val_dataset = HeartSoundDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class BalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that balances class distribution in each batch.
    
    Important for heart sound classification where abnormal samples
    are often underrepresented.
    """
    
    def __init__(self, labels: np.ndarray):
        self.labels = labels
        self.indices_per_class = {
            0: np.where(labels == 0)[0],
            1: np.where(labels == 1)[0],
        }
        
        # Oversample minority class
        self.num_samples = 2 * max(
            len(self.indices_per_class[0]),
            len(self.indices_per_class[1])
        )
    
    def __iter__(self):
        indices = []
        for _ in range(self.num_samples // 2):
            # Sample one from each class
            idx_0 = np.random.choice(self.indices_per_class[0])
            idx_1 = np.random.choice(self.indices_per_class[1])
            indices.extend([idx_0, idx_1])
        
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing HeartSoundDataset...")
    
    # Create dummy data
    dummy_specs = np.random.randn(100, 128, 78)  # 100 samples, 128 mel bins, 78 time frames
    dummy_labels = np.random.randint(0, 2, 100)
    
    dataset = HeartSoundDataset(dummy_specs, dummy_labels, augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Label: {label}")
    
    # Test data loader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_specs, batch_labels = next(iter(loader))
    print(f"Batch shape: {batch_specs.shape}")
    print(f"Batch labels: {batch_labels}")
