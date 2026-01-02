"""
Heart Sound Classification Pipeline Visualization
==================================================
Creates a visual diagram of the complete heart sound classification pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def create_pipeline_diagram():
    """Create a comprehensive pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    colors = {
        'input': '#E8F4F8',
        'preprocess': '#FFF4E6',
        'feature': '#E8F5E9',
        'model': '#F3E5F5',
        'output': '#FFEBEE',
        'border': '#37474F',
        'text': '#263238',
        'arrow': '#546E7A'
    }

    # Title
    ax.text(8, 9.5, 'Heart Sound Classification Pipeline',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color=colors['text'])

    # Define stages
    stages = [
        {
            'name': 'Input',
            'x': 0.5, 'y': 7.5, 'width': 2.5, 'height': 1.5,
            'color': colors['input'],
            'items': ['Raw Audio (.wav)', 'PhysioNet Dataset', '2000 Hz Sample Rate']
        },
        {
            'name': 'Preprocessing',
            'x': 3.5, 'y': 7.5, 'width': 2.5, 'height': 1.5,
            'color': colors['preprocess'],
            'items': ['Bandpass Filter', '(25-400 Hz)', 'Amplitude Normalization']
        },
        {
            'name': 'Segmentation',
            'x': 6.5, 'y': 7.5, 'width': 2.5, 'height': 1.5,
            'color': colors['preprocess'],
            'items': ['5-sec Windows', '50% Overlap', 'Padding for Short Audio']
        },
        {
            'name': 'Feature Extraction',
            'x': 9.5, 'y': 7.5, 'width': 2.5, 'height': 1.5,
            'color': colors['feature'],
            'items': ['Mel Spectrogram', '128 Mel Bins', 'Log Scale (dB)']
        },
        {
            'name': 'Data Augmentation',
            'x': 12.5, 'y': 7.5, 'width': 2.5, 'height': 1.5,
            'color': colors['feature'],
            'items': ['Time Shifting', 'Frequency Masking', 'Amplitude Scaling']
        }
    ]

    # Draw main pipeline stages
    for i, stage in enumerate(stages):
        # Box
        box = FancyBboxPatch(
            (stage['x'], stage['y']),
            stage['width'], stage['height'],
            boxstyle="round,pad=0.1",
            edgecolor=colors['border'],
            facecolor=stage['color'],
            linewidth=2
        )
        ax.add_patch(box)

        # Title
        ax.text(stage['x'] + stage['width']/2, stage['y'] + stage['height'] - 0.3,
                stage['name'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=colors['text'])

        # Items
        for j, item in enumerate(stage['items']):
            ax.text(stage['x'] + stage['width']/2,
                    stage['y'] + stage['height'] - 0.6 - j*0.35,
                    item, ha='center', va='center',
                    fontsize=9, color=colors['text'])

        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch(
                (stage['x'] + stage['width'], stage['y'] + stage['height']/2),
                (stages[i+1]['x'], stages[i+1]['y'] + stages[i+1]['height']/2),
                arrowstyle='->', mutation_scale=20,
                color=colors['arrow'], linewidth=2
            )
            ax.add_patch(arrow)

    # Model section
    model_box = FancyBboxPatch(
        (2, 4.5), 12, 2,
        boxstyle="round,pad=0.1",
        edgecolor=colors['border'],
        facecolor=colors['model'],
        linewidth=2
    )
    ax.add_patch(model_box)

    ax.text(8, 6.3, 'Deep Learning Model', ha='center', va='center',
            fontsize=12, fontweight='bold', color=colors['text'])

    # Model options
    models = [
        {'name': 'Custom CNN', 'params': '~500K', 'acc': '75-80%', 'x': 3},
        {'name': 'CNN Light', 'params': '~200K', 'acc': '70-75%', 'x': 5.5},
        {'name': 'ResNet18', 'params': '~11M', 'acc': '80-85%', 'x': 8},
        {'name': 'CNN + Attention', 'params': '~600K', 'acc': '78-83%', 'x': 10.5}
    ]

    for model in models:
        # Model box
        m_box = FancyBboxPatch(
            (model['x'], 4.8), 2, 1.2,
            boxstyle="round,pad=0.05",
            edgecolor=colors['border'],
            facecolor='white',
            linewidth=1.5
        )
        ax.add_patch(m_box)

        ax.text(model['x'] + 1, 5.7, model['name'], ha='center', va='center',
                fontsize=10, fontweight='bold', color=colors['text'])
        ax.text(model['x'] + 1, 5.4, f"{model['params']} params", ha='center', va='center',
                fontsize=8, color=colors['text'])
        ax.text(model['x'] + 1, 5.1, f"Acc: {model['acc']}", ha='center', va='center',
                fontsize=8, color=colors['text'])

    # Arrow from augmentation to model
    arrow = FancyArrowPatch(
        (13.75, 8.25), (8, 6.5),
        arrowstyle='->', mutation_scale=20,
        color=colors['arrow'], linewidth=2,
        connectionstyle="arc3,rad=-0.3"
    )
    ax.add_patch(arrow)

    # Output section
    output_box = FancyBboxPatch(
        (5, 2), 6, 1.5,
        boxstyle="round,pad=0.1",
        edgecolor=colors['border'],
        facecolor=colors['output'],
        linewidth=2
    )
    ax.add_patch(output_box)

    ax.text(8, 3.2, 'Classification Output', ha='center', va='center',
            fontsize=12, fontweight='bold', color=colors['text'])

    ax.text(8, 2.7, 'Normal (0) vs Abnormal (1)', ha='center', va='center',
            fontsize=10, color=colors['text'])
    ax.text(8, 2.3, 'Probability Score + Confidence', ha='center', va='center',
            fontsize=9, color=colors['text'])

    # Arrow from model to output
    arrow = FancyArrowPatch(
        (8, 4.5), (8, 3.5),
        arrowstyle='->', mutation_scale=20,
        color=colors['arrow'], linewidth=2
    )
    ax.add_patch(arrow)

    # Training details box
    train_box = FancyBboxPatch(
        (0.5, 0.2), 15, 1.3,
        boxstyle="round,pad=0.1",
        edgecolor=colors['border'],
        facecolor='#F5F5F5',
        linewidth=2
    )
    ax.add_patch(train_box)

    ax.text(8, 1.3, 'Training Pipeline Details', ha='center', va='center',
            fontsize=11, fontweight='bold', color=colors['text'])

    train_details = [
        'Loss: Cross-Entropy (with class weights)',
        'Optimizer: Adam with ReduceLROnPlateau scheduler',
        'Metrics: Accuracy, ROC-AUC, Confusion Matrix',
        'Validation: 20% holdout with stratified split'
    ]

    for i, detail in enumerate(train_details):
        ax.text(8, 1.0 - i*0.2, detail, ha='center', va='center',
                fontsize=9, color=colors['text'])

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Data'),
        mpatches.Patch(color=colors['preprocess'], label='Preprocessing'),
        mpatches.Patch(color=colors['feature'], label='Features'),
        mpatches.Patch(color=colors['model'], label='Model'),
        mpatches.Patch(color=colors['output'], label='Output')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(15.5, 9.5), fontsize=9)

    plt.tight_layout()
    return fig

def create_detailed_preprocessing_diagram():
    """Create a detailed preprocessing pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Detailed Preprocessing Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold')

    # Preprocessing steps
    steps = [
        {
            'name': '1. Load Audio',
            'x': 0.5, 'y': 5.5, 'width': 2.5, 'height': 1.5,
            'details': ['Load .wav file', 'Resample to 2000 Hz', 'Convert to mono']
        },
        {
            'name': '2. Bandpass Filter',
            'x': 3.5, 'y': 5.5, 'width': 2.5, 'height': 1.5,
            'details': ['Butterworth 5th order', '25-400 Hz range', 'filtfilt (zero-phase)']
        },
        {
            'name': '3. Normalize',
            'x': 6.5, 'y': 5.5, 'width': 2.5, 'height': 1.5,
            'details': ['Scale to [-1, 1]', 'Peak normalization', 'Preserve dynamics']
        },
        {
            'name': '4. Segment',
            'x': 9.5, 'y': 5.5, 'width': 2.5, 'height': 1.5,
            'details': ['5-second windows', '50% overlap', 'Zero-pad short audio']
        },
        {
            'name': '5. Mel Spectrogram',
            'x': 12.5, 'y': 5.5, 'width': 1.5, 'height': 1.5,
            'details': ['128 mel bins', 'FFT: 512', 'Hop: 128', 'Log scale (dB)']
        }
    ]

    colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5']

    for i, step in enumerate(steps):
        # Box
        box = FancyBboxPatch(
            (step['x'], step['y']),
            step['width'], step['height'],
            boxstyle="round,pad=0.1",
            edgecolor='#1565C0',
            facecolor=colors[i],
            linewidth=2
        )
        ax.add_patch(box)

        # Title
        ax.text(step['x'] + step['width']/2, step['y'] + step['height'] - 0.3,
                step['name'], ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Details
        for j, detail in enumerate(step['details']):
            ax.text(step['x'] + step['width']/2,
                    step['y'] + step['height'] - 0.55 - j*0.3,
                    detail, ha='center', va='center',
                    fontsize=8)

        # Arrow to next step
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (step['x'] + step['width'], step['y'] + step['height']/2),
                (steps[i+1]['x'], steps[i+1]['y'] + steps[i+1]['height']/2),
                arrowstyle='->', mutation_scale=15,
                color='#1565C0', linewidth=2
            )
            ax.add_patch(arrow)

    # Signal processing explanation
    explanation_box = FancyBboxPatch(
        (0.5, 1), 13, 3,
        boxstyle="round,pad=0.1",
        edgecolor='#424242',
        facecolor='#FAFAFA',
        linewidth=2
    )
    ax.add_patch(explanation_box)

    ax.text(7, 3.7, 'Why This Pipeline Works', ha='center', va='center',
            fontsize=12, fontweight='bold')

    explanations = [
        '• Heart sounds (S1, S2) primarily occupy 25-400 Hz frequency range',
        '• Bandpass filtering removes baseline wander, breathing, and high-frequency noise',
        '• Mel scale approximates human auditory perception for better feature extraction',
        '• Spectrograms preserve temporal and frequency patterns that CNNs can learn',
        '• Segmentation increases training data and handles variable-length recordings',
        '• Log-scale compression makes features more robust to amplitude variations'
    ]

    for i, exp in enumerate(explanations):
        ax.text(1, 3.3 - i*0.35, exp, ha='left', va='center',
                fontsize=9)

    plt.tight_layout()
    return fig

def create_model_architecture_diagram():
    """Create a diagram showing the CNN architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(7, 8.5, 'Custom CNN Architecture', ha='center', va='center',
            fontsize=18, fontweight='bold')

    # Input
    input_box = FancyBboxPatch(
        (0.5, 6.5), 2, 1.2,
        boxstyle="round,pad=0.1",
        edgecolor='#2E7D32',
        facecolor='#C8E6C9',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(1.5, 7.3, 'Input', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 7.0, '(1, 128, 78)', ha='center', va='center', fontsize=9)

    # Conv blocks
    conv_blocks = [
        {'name': 'Conv Block 1', 'channels': '32', 'x': 3.5},
        {'name': 'Conv Block 2', 'channels': '64', 'x': 5.5},
        {'name': 'Conv Block 3', 'channels': '128', 'x': 7.5},
        {'name': 'Conv Block 4', 'channels': '256', 'x': 9.5}
    ]

    for i, block in enumerate(conv_blocks):
        box = FancyBboxPatch(
            (block['x'], 6.5), 1.5, 1.2,
            boxstyle="round,pad=0.1",
            edgecolor='#1565C0',
            facecolor='#BBDEFB',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(block['x'] + 0.75, 7.2, block['name'], ha='center', va='center',
                fontsize=9, fontweight='bold')
        ax.text(block['x'] + 0.75, 6.9, f'{block["channels"]} ch', ha='center', va='center',
                fontsize=8)

        # Arrow
        if i == 0:
            arrow = FancyArrowPatch(
                (2.5, 7.1), (3.5, 7.1),
                arrowstyle='->', mutation_scale=15,
                color='#424242', linewidth=2
            )
        else:
            arrow = FancyArrowPatch(
                (conv_blocks[i-1]['x'] + 1.5, 7.1), (block['x'], 7.1),
                arrowstyle='->', mutation_scale=15,
                color='#424242', linewidth=2
            )
        ax.add_patch(arrow)

    # Global pooling
    pool_box = FancyBboxPatch(
        (11.5, 6.5), 1.5, 1.2,
        boxstyle="round,pad=0.1",
        edgecolor='#6A1B9A',
        facecolor='#E1BEE7',
        linewidth=2
    )
    ax.add_patch(pool_box)
    ax.text(12.25, 7.2, 'Global', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(12.25, 6.9, 'Avg Pool', ha='center', va='center', fontsize=9)

    arrow = FancyArrowPatch(
        (11, 7.1), (11.5, 7.1),
        arrowstyle='->', mutation_scale=15,
        color='#424242', linewidth=2
    )
    ax.add_patch(arrow)

    # Classifier
    classifier_box = FancyBboxPatch(
        (5, 4.5), 4, 1.5,
        boxstyle="round,pad=0.1",
        edgecolor='#C62828',
        facecolor='#FFCDD2',
        linewidth=2
    )
    ax.add_patch(classifier_box)
    ax.text(7, 5.7, 'Classifier', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, 5.3, 'FC: 256 → 128 → 64 → 2', ha='center', va='center', fontsize=9)
    ax.text(7, 4.9, 'Dropout: 0.5', ha='center', va='center', fontsize=9)

    arrow = FancyArrowPatch(
        (12.25, 6.5), (7, 6),
        arrowstyle='->', mutation_scale=15,
        color='#424242', linewidth=2,
        connectionstyle="arc3,rad=-0.3"
    )
    ax.add_patch(arrow)

    # Output
    output_box = FancyBboxPatch(
        (5.5, 2.5), 3, 1.2,
        boxstyle="round,pad=0.1",
        edgecolor='#BF360C',
        facecolor='#FFCCBC',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(7, 3.2, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7, 2.9, 'Normal / Abnormal', ha='center', va='center', fontsize=9)

    arrow = FancyArrowPatch(
        (7, 4.5), (7, 3.7),
        arrowstyle='->', mutation_scale=15,
        color='#424242', linewidth=2
    )
    ax.add_patch(arrow)

    # Conv block details
    details_box = FancyBboxPatch(
        (0.5, 0.2), 13, 1.8,
        boxstyle="round,pad=0.1",
        edgecolor='#424242',
        facecolor='#F5F5F5',
        linewidth=2
    )
    ax.add_patch(details_box)

    ax.text(7, 1.8, 'Convolutional Block Details', ha='center', va='center',
            fontsize=11, fontweight='bold')

    details = [
        'Each block: Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU → MaxPool2d(2×2)',
        'Kaiming weight initialization | Global average pooling handles variable-length input',
        'Total parameters: ~500K | Training time: 5-10 min (CPU) | Accuracy: 75-80%'
    ]

    for i, detail in enumerate(details):
        ax.text(7, 1.4 - i*0.35, detail, ha='center', va='center', fontsize=9)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create all diagrams
    print("Creating pipeline diagrams...")

    # Main pipeline diagram
    fig1 = create_pipeline_diagram()
    fig1.savefig('/Users/leosakharov/dev/heart_sound_classifier/pipeline_diagram.png',
                 dpi=150, bbox_inches='tight')
    print("✓ Saved pipeline_diagram.png")

    # Detailed preprocessing diagram
    fig2 = create_detailed_preprocessing_diagram()
    fig2.savefig('/Users/leosakharov/dev/heart_sound_classifier/preprocessing_diagram.png',
                 dpi=150, bbox_inches='tight')
    print("✓ Saved preprocessing_diagram.png")

    # Model architecture diagram
    fig3 = create_model_architecture_diagram()
    fig3.savefig('/Users/leosakharov/dev/heart_sound_classifier/model_architecture.png',
                 dpi=150, bbox_inches='tight')
    print("✓ Saved model_architecture.png")

    plt.close('all')
    print("\n✅ All diagrams created successfully!")
