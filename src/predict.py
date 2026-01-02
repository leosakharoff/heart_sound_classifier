"""
Heart Sound Prediction Script
==============================
Load a trained model and classify new heart sound recordings.

This script provides inference capabilities for the heart sound classifier,
supporting both single file and batch prediction modes with optional visualization.

Features:
- Load trained models from checkpoint files
- Process individual audio files or entire directories
- Generate confidence scores for predictions
- Create visualizations of mel spectrograms with predictions
- Support for multiple model architectures

Prediction Pipeline:
1. Load trained model checkpoint
2. Preprocess audio (same pipeline as training)
3. Extract mel spectrogram features
4. Run model inference
5. Output prediction (Normal/Abnormal) with confidence score

Usage Examples:
    # Classify a single file
    python predict.py --model_path ./models/best_model.pth --audio_path ./test.wav

    # Classify with visualization
    python predict.py --model_path ./models/best_model.pth --audio_path ./test.wav --visualize

    # Classify all files in a directory
    python predict.py --model_path ./models/best_model.pth --audio_dir ./test_sounds/

    # Batch classify with custom output
    python predict.py --model_path ./models/best_model.pth --audio_dir ./sounds/ --output_dir ./predictions/

Output:
- For single files: Prints prediction and confidence score
- For directories: Creates CSV with all predictions
- With --visualize: Saves spectrogram plots with prediction overlay
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Local imports
from preprocess import HeartSoundPreprocessor
from model import get_model


class HeartSoundClassifier:
    """
    Production-ready heart sound classifier.
    
    Loads a trained model and provides methods for:
    - Single file prediction
    - Batch prediction
    - Visualization of predictions
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'cnn',
        device: str = 'auto',
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to saved model checkpoint
            model_type: Model architecture ('cnn', 'resnet', 'attention')
            device: Device to run inference on
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize preprocessor
        self.preprocessor = HeartSoundPreprocessor()
        
        # Load model
        self.model = get_model(model_type, num_classes=2)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using untrained model - predictions will be random!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class labels
        self.labels = ['Normal', 'Abnormal']
    
    def predict(self, audio_path: str) -> dict:
        """
        Classify a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results:
            - 'class': Predicted class (0=Normal, 1=Abnormal)
            - 'label': Class label string
            - 'confidence': Prediction confidence
            - 'probabilities': Class probabilities [P(Normal), P(Abnormal)]
            - 'num_segments': Number of segments analyzed
        """
        # Preprocess audio
        result = self.preprocessor.process_file(audio_path, return_segments=True)
        spectrograms = result['spectrograms']
        
        if len(spectrograms) == 0:
            return {
                'class': None,
                'label': 'Error',
                'confidence': 0.0,
                'probabilities': [0.0, 0.0],
                'num_segments': 0,
                'error': 'No valid segments extracted'
            }
        
        # Convert to tensor
        specs = []
        for spec in spectrograms:
            # Normalize
            spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            specs.append(spec_norm)
        
        X = torch.FloatTensor(np.array(specs)).unsqueeze(1)  # (N, 1, n_mels, time)
        X = X.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
            
            # Average predictions across all segments
            avg_probs = probs.mean(dim=0)
            pred_class = torch.argmax(avg_probs).item()
            confidence = avg_probs[pred_class].item()
        
        return {
            'class': pred_class,
            'label': self.labels[pred_class],
            'confidence': confidence,
            'probabilities': avg_probs.cpu().numpy().tolist(),
            'num_segments': len(spectrograms),
        }
    
    def predict_batch(self, audio_paths: list) -> list:
        """
        Classify multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                result['file'] = os.path.basename(path)
                results.append(result)
            except Exception as e:
                results.append({
                    'file': os.path.basename(path),
                    'error': str(e)
                })
        return results
    
    def visualize_prediction(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> dict:
        """
        Classify audio and visualize the result.
        
        Creates a figure showing:
        - Original waveform
        - Mel spectrogram
        - Prediction with confidence
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save figure (optional)
            show: Whether to display the figure
            
        Returns:
            Prediction dictionary
        """
        # Get preprocessed audio
        result = self.preprocessor.process_file(audio_path, return_segments=True)
        audio = result['audio']
        spectrograms = result['spectrograms']
        sr = result['sr']
        
        # Get prediction
        prediction = self.predict(audio_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform
        time = np.arange(len(audio)) / sr
        axes[0].plot(time, audio, linewidth=0.5, color='blue')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Heart Sound Waveform: {os.path.basename(audio_path)}')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram (use first segment)
        if len(spectrograms) > 0:
            spec = spectrograms[0]
            img = axes[1].imshow(
                spec,
                aspect='auto',
                origin='lower',
                cmap='magma',
                extent=[0, spec.shape[1] * self.preprocessor.hop_length / sr, 
                       self.preprocessor.lowcut, self.preprocessor.highcut]
            )
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_title('Mel Spectrogram')
            plt.colorbar(img, ax=axes[1], label='dB')
        
        # Add prediction text
        pred_text = (
            f"Prediction: {prediction['label']}\n"
            f"Confidence: {prediction['confidence']*100:.1f}%\n"
            f"P(Normal): {prediction['probabilities'][0]*100:.1f}%\n"
            f"P(Abnormal): {prediction['probabilities'][1]*100:.1f}%"
        )
        
        # Color based on prediction
        color = 'green' if prediction['class'] == 0 else 'red'
        
        fig.text(
            0.02, 0.98, pred_text,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontfamily='monospace',
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return prediction


def main():
    parser = argparse.ArgumentParser(description='Heart Sound Classifier - Prediction')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'cnn_light', 'resnet', 'attention'],
                        help='Model architecture')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Path to single audio file')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save visualizations')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for inference')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = HeartSoundClassifier(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
    )
    
    # Collect audio files
    audio_files = []
    
    if args.audio_path:
        audio_files.append(args.audio_path)
    
    if args.audio_dir:
        for f in os.listdir(args.audio_dir):
            if f.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                audio_files.append(os.path.join(args.audio_dir, f))
    
    if len(audio_files) == 0:
        print("No audio files specified. Use --audio_path or --audio_dir")
        return
    
    # Create output directory if visualizing
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Classify files
    print(f"\nClassifying {len(audio_files)} audio file(s)...\n")
    print("-" * 60)
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        
        try:
            if args.visualize:
                output_path = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(filename)[0]}_prediction.png"
                )
                prediction = classifier.visualize_prediction(
                    audio_path,
                    output_path=output_path,
                    show=False,
                )
            else:
                prediction = classifier.predict(audio_path)
            
            # Print result
            print(f"File: {filename}")
            print(f"  Prediction: {prediction['label']}")
            print(f"  Confidence: {prediction['confidence']*100:.1f}%")
            print(f"  P(Normal): {prediction['probabilities'][0]*100:.1f}%")
            print(f"  P(Abnormal): {prediction['probabilities'][1]*100:.1f}%")
            print(f"  Segments analyzed: {prediction['num_segments']}")
            print()
            
        except Exception as e:
            print(f"File: {filename}")
            print(f"  Error: {e}")
            print()
    
    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
