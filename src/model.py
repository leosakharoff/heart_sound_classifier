"""
CNN architectures for heart sound classification.
Includes lightweight CNN, ResNet18, and attention-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class HeartSoundCNN(nn.Module):
    """4-layer CNN for spectrogram classification. Fast to train, ~500K params."""
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout: float = 0.5,
    ):
        super(HeartSoundCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = self._conv_block(input_channels, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class HeartSoundResNet(nn.Module):
    """ResNet18 adapted for single-channel spectrograms. Higher accuracy, ~11M params."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super(HeartSoundResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for single-channel input
        # Original: Conv2d(3, 64, 7, 2, 3)
        # Modified: Conv2d(1, 64, 7, 2, 3)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new conv layer with mean of RGB channels
        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )
        
        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for name, param in self.resnet.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class AttentionBlock(nn.Module):
    """Learns to focus on important time-frequency regions."""
    
    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class HeartSoundCNNLight(nn.Module):
    """Smaller 3-layer CNN. Good for small datasets, ~100K params."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout: float = 0.5,
    ):
        super(HeartSoundCNNLight, self).__init__()

        # Convolutional blocks (3 blocks vs 4 in full CNN)
        self.conv1 = self._conv_block(input_channels, 16)   # 16 vs 32
        self.conv2 = self._conv_block(16, 32)              # 32 vs 64
        self.conv3 = self._conv_block(32, 64)              # 64 vs 128

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simpler classifier (no intermediate layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class HeartSoundCNNWithAttention(nn.Module):
    """CNN with attention to focus on relevant heart sound events."""
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout: float = 0.5,
    ):
        super(HeartSoundCNNWithAttention, self).__init__()
        
        # Encoder
        self.conv1 = self._conv_block(input_channels, 32)
        self.conv2 = self._conv_block(32, 64)
        self.attention = AttentionBlock(64)
        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)  # Apply attention
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def get_model(
    model_type: str = 'cnn',
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Get a model by name: 'cnn', 'cnn_light', 'resnet', or 'attention'."""
    if model_type == 'cnn':
        return HeartSoundCNN(num_classes=num_classes)
    elif model_type == 'cnn_light':
        return HeartSoundCNNLight(num_classes=num_classes)
    elif model_type == 'resnet':
        return HeartSoundResNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'attention':
        return HeartSoundCNNWithAttention(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total params, trainable params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test all models
    print("Testing model architectures...\n")
    
    # Create dummy input (batch=4, channels=1, n_mels=128, time_frames=78)
    dummy_input = torch.randn(4, 1, 128, 78)

    for model_type in ['cnn', 'cnn_light', 'resnet', 'attention']:
        print(f"--- {model_type.upper()} ---")
        model = get_model(model_type)
        
        # Forward pass
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Parameter count
        total, trainable = count_parameters(model)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print()
