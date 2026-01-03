"""
Trains CNN models to classify heart sounds as normal or abnormal.
Loads PhysioNet 2016 data, preprocesses audio, and saves the best model.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Local imports
from preprocess import HeartSoundPreprocessor, preprocess_dataset
from dataset import HeartSoundDataset, create_data_loaders
from model import get_model, count_parameters


def load_physionet_data(data_dir: str, max_files: int = None):
    """Load audio files and labels from PhysioNet 2016 dataset."""
    audio_files = []
    labels = []
    
    # PhysioNet 2016 training sets
    training_sets = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
    
    for training_set in training_sets:
        set_dir = os.path.join(data_dir, training_set)
        
        if not os.path.exists(set_dir):
            continue
        
        # Read reference file
        ref_file = os.path.join(set_dir, 'REFERENCE.csv')
        if not os.path.exists(ref_file):
            # Try alternative name
            ref_file = os.path.join(set_dir, 'REFERENCE.txt')
        
        if os.path.exists(ref_file):
            with open(ref_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename = parts[0]
                        label = int(parts[1])
                        
                        # Convert label: -1 (normal) -> 0, 1 (abnormal) -> 1
                        label = 0 if label == -1 else 1
                        
                        audio_path = os.path.join(set_dir, f"{filename}.wav")
                        if os.path.exists(audio_path):
                            audio_files.append(audio_path)
                            labels.append(label)
    
    # Limit files if specified
    if max_files and len(audio_files) > max_files:
        indices = np.random.choice(len(audio_files), max_files, replace=False)
        audio_files = [audio_files[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    print(f"Loaded {len(audio_files)} audio files")
    print(f"Class distribution: Normal={sum(1 for l in labels if l==0)}, Abnormal={sum(1 for l in labels if l==1)}")
    
    return audio_files, labels


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def train(
    data_dir: str,
    output_dir: str = './models',
    model_type: str = 'cnn',
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_files: int = None,
    device: str = 'auto',
):
    """Train the model and save results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    audio_files, labels = load_physionet_data(data_dir, max_files)
    
    if len(audio_files) == 0:
        print("No data found! Please check the data directory.")
        print(f"Expected structure: {data_dir}/training-a/, training-b/, etc.")
        return
    
    # Preprocess audio
    print("\n2. Preprocessing audio...")
    preprocessor = HeartSoundPreprocessor()
    X, y = preprocess_dataset(audio_files, labels, preprocessor)
    
    # Train/validation split
    print("\n3. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create data loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=batch_size,
    )
    
    # Create model
    print(f"\n5. Creating {model_type} model...")
    model = get_model(model_type)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n6. Training...")
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  -> Saved new best model (acc: {val_acc:.4f})")
    
    # Final evaluation
    print("\n7. Final Evaluation...")
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, final_preds, final_labels, final_probs = validate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        final_labels, final_preds,
        target_names=['Normal', 'Abnormal']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC
    try:
        auc = roc_auc_score(final_labels, final_probs)
        print(f"\nROC AUC Score: {auc:.4f}")
    except:
        auc = None
    
    # Save results
    results = {
        'model_type': model_type,
        'epochs': epochs,
        'best_val_acc': best_val_acc,
        'final_accuracy': np.mean(np.array(final_preds) == np.array(final_labels)),
        'roc_auc': auc,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pth")
    
    return model, history


def plot_training_curves(history: dict, output_dir: str):
    """Save loss and accuracy plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Heart Sound Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to PhysioNet 2016 data directory')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save model and results')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'cnn_light', 'resnet', 'attention'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to load (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to train on')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_files=args.max_files,
        device=args.device,
    )
