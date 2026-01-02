"""
Heart Sound Preprocessing Module
================================
Comprehensive audio preprocessing pipeline for heart sound classification.

This module handles the entire signal processing pipeline from raw audio
recordings to mel spectrogram features ready for deep learning models.

Key Processing Steps:
1. Audio Loading: Load WAV files and resample to 2000 Hz (standard for heart sounds)
2. Bandpass Filtering: Apply 5th-order Butterworth filter (25-400 Hz)
   - Removes low-frequency baseline wander and breathing sounds
   - Removes high-frequency noise and interference
   - Preserves S1, S2 heartbeats and murmurs
3. Amplitude Normalization: Scale to [-1, 1] range
4. Mel Spectrogram Extraction: Convert to time-frequency representation
   - 128 mel frequency bins
   - Captures perceptually relevant features
5. Segmentation: Split into fixed-length windows (5 seconds, 50% overlap)

Why This Pipeline Works:
- Heart sounds primarily occupy 25-400 Hz frequency range
- Mel scale approximates human auditory perception
- Spectrograms preserve temporal and frequency patterns
- CNNs excel at learning patterns from spectrogram representations

Usage:
    from preprocess import HeartSoundPreprocessor, preprocess_dataset

    # Preprocess a single file
    preprocessor = HeartSoundPreprocessor()
    spectrogram = preprocessor.process('heart_sound.wav')

    # Preprocess multiple files
    spectrograms, labels = preprocess_dataset(audio_files, labels, preprocessor)
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HeartSoundPreprocessor:
    """
    Preprocesses heart sound recordings for classification.
    
    Pipeline:
    1. Load audio and resample to target sample rate
    2. Apply bandpass filter to isolate heart sound frequencies
    3. Normalize amplitude
    4. Extract mel spectrogram features
    """
    
    def __init__(
        self,
        target_sr: int = 2000,
        lowcut: float = 25.0,
        highcut: float = 400.0,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 128,
        segment_duration: float = 5.0,
    ):
        """
        Initialize preprocessor with audio parameters.
        
        Args:
            target_sr: Target sample rate (2000 Hz standard for heart sounds)
            lowcut: Lower cutoff frequency for bandpass filter
            highcut: Upper cutoff frequency for bandpass filter
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for spectrogram
            segment_duration: Duration of each segment in seconds
        """
        self.target_sr = target_sr
        self.lowcut = lowcut
        self.highcut = highcut
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * target_sr)
        
        # Pre-compute filter coefficients
        self._compute_filter_coefficients()
    
    def _compute_filter_coefficients(self):
        """Compute Butterworth bandpass filter coefficients."""
        nyquist = 0.5 * self.target_sr
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        self.b, self.a = butter(5, [low, high], btype='band')
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            
        Returns:
            Tuple of (audio signal, sample rate)
        """
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return y, sr
    
    def bandpass_filter(self, y: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to isolate heart sound frequencies.
        
        Heart sounds (S1, S2) are typically in 20-150 Hz range.
        Murmurs can extend up to 400 Hz.
        This filter removes:
        - Low-frequency baseline wander
        - High-frequency noise and artifacts
        
        Args:
            y: Audio signal
            
        Returns:
            Filtered audio signal
        """
        # Handle edge case of very short signals
        if len(y) < 50:
            return y
            
        try:
            y_filtered = filtfilt(self.b, self.a, y)
        except ValueError:
            # If filtfilt fails, return original signal
            return y
            
        return y_filtered
    
    def normalize(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            y: Audio signal
            
        Returns:
            Normalized audio signal
        """
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
        return y
    
    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram from audio signal.
        
        The mel scale approximates human auditory perception,
        making it effective for audio classification tasks.
        
        Args:
            y: Audio signal
            
        Returns:
            Log-mel spectrogram (n_mels x time_frames)
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.target_sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.lowcut,
            fmax=self.highcut,
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def segment_audio(
        self, 
        y: np.ndarray, 
        overlap: float = 0.5
    ) -> list:
        """
        Segment audio into fixed-length windows.
        
        This ensures consistent input size for the neural network
        and increases training data through overlapping windows.
        
        Args:
            y: Audio signal
            overlap: Overlap fraction between segments (0.5 = 50%)
            
        Returns:
            List of audio segments
        """
        segments = []
        step = int(self.segment_samples * (1 - overlap))
        
        for start in range(0, len(y) - self.segment_samples + 1, step):
            segment = y[start:start + self.segment_samples]
            segments.append(segment)
        
        # Handle case where audio is shorter than segment duration
        if len(segments) == 0 and len(y) > 0:
            # Pad short audio with zeros
            padded = np.zeros(self.segment_samples)
            padded[:len(y)] = y
            segments.append(padded)
        
        return segments
    
    def process_file(
        self, 
        audio_path: str,
        return_segments: bool = True
    ) -> dict:
        """
        Complete preprocessing pipeline for a single audio file.
        
        Args:
            audio_path: Path to audio file
            return_segments: If True, return segmented spectrograms
            
        Returns:
            Dictionary containing:
            - 'spectrograms': List of mel spectrograms
            - 'audio': Preprocessed audio signal
            - 'sr': Sample rate
        """
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        # Apply bandpass filter
        y_filtered = self.bandpass_filter(y)
        
        # Normalize
        y_normalized = self.normalize(y_filtered)
        
        if return_segments:
            # Segment and extract spectrograms
            segments = self.segment_audio(y_normalized)
            spectrograms = [self.extract_mel_spectrogram(seg) for seg in segments]
        else:
            # Single spectrogram for entire recording
            spectrograms = [self.extract_mel_spectrogram(y_normalized)]
        
        return {
            'spectrograms': spectrograms,
            'audio': y_normalized,
            'sr': sr,
        }


def preprocess_dataset(
    audio_files: list,
    labels: list,
    preprocessor: HeartSoundPreprocessor,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess entire dataset of audio files.
    
    Args:
        audio_files: List of paths to audio files
        labels: List of labels (0=normal, 1=abnormal)
        preprocessor: HeartSoundPreprocessor instance
        verbose: Print progress
        
    Returns:
        Tuple of (spectrograms array, labels array)
    """
    all_spectrograms = []
    all_labels = []
    
    for i, (audio_path, label) in enumerate(zip(audio_files, labels)):
        if verbose and i % 50 == 0:
            print(f"Processing file {i+1}/{len(audio_files)}...")
        
        try:
            result = preprocessor.process_file(audio_path)
            for spec in result['spectrograms']:
                all_spectrograms.append(spec)
                all_labels.append(label)
        except Exception as e:
            if verbose:
                print(f"Error processing {audio_path}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.array(all_spectrograms)
    y = np.array(all_labels)
    
    if verbose:
        print(f"Dataset processed: {len(X)} samples")
        print(f"Spectrogram shape: {X[0].shape}")
        print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    
    return X, y


if __name__ == "__main__":
    # Quick test
    print("Heart Sound Preprocessor initialized successfully!")
    
    preprocessor = HeartSoundPreprocessor()
    print(f"Target sample rate: {preprocessor.target_sr} Hz")
    print(f"Bandpass filter: {preprocessor.lowcut}-{preprocessor.highcut} Hz")
    print(f"Mel bins: {preprocessor.n_mels}")
    print(f"Segment duration: {preprocessor.segment_duration}s")
