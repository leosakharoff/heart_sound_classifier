"""
Audio preprocessing for heart sound classification.
Handles filtering, normalization, and mel spectrogram extraction.
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HeartSoundPreprocessor:
    """Converts raw audio to mel spectrograms for the classifier."""
    
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
        nyquist = 0.5 * self.target_sr
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        self.b, self.a = butter(5, [low, high], btype='band')
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and resample audio file."""
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return y, sr
    
    def bandpass_filter(self, y: np.ndarray) -> np.ndarray:
        """Filter to keep only heart sound frequencies (25-400 Hz)."""
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
        """Scale audio to [-1, 1] range."""
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
        return y
    
    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Convert audio to log-mel spectrogram."""
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
        """Split audio into fixed-length overlapping windows."""
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
        """Run full preprocessing pipeline on an audio file."""
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
    """Preprocess a list of audio files and return spectrograms with labels."""
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
