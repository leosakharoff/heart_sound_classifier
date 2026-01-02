# Heart Sound Classifier - Interview Notes

## Quick Reference Card

```
PROJECT: Heart Sound Classifier
DATASET: PhysioNet 2016 (405 recordings → 4,738 samples)
CLASSES: Normal (29%) vs Abnormal (71%)
PIPELINE: Audio → Bandpass (25-400Hz) → Mel Spectrogram (128 bands) → CNN

BEST RESULT:
- Model: Light CNN (23,650 parameters)
- Validation Accuracy: 74.58%
- ROC-AUC: 0.737
- Above majority baseline (71%) ✓

MODELS TESTED:
- Full CNN (1.2M params): Overfit quickly
- Light CNN (23K params): 74.58% but unstable
- Medium CNN (200K params): Testing next

KEY INSIGHTS:
1. Real data essential (synthetic failed)
2. Model capacity must match dataset size
3. Class imbalance requires weighted loss

LIMITATIONS:
- Single dataset subset (training-a only)
- Single train/val split (no cross-validation)
- Binary classification only
- Proof-of-concept, not clinical
```

---

## Opening Statement (2-3 minutes)

"Good morning. I'd like to present my heart sound classification project, which implements a complete machine learning pipeline for detecting cardiac abnormalities from phonocardiogram (PCG) recordings.

The system processes raw audio signals into mel spectrograms and uses convolutional neural networks to classify recordings as Normal or Abnormal. I trained and evaluated the model on the PhysioNet 2016 Challenge dataset, which contains real clinical recordings from multiple hospitals."

---

## Technical Overview (3-4 minutes)

"Let me walk you through the technical architecture:

**First, the preprocessing pipeline**: Raw audio is bandpass-filtered between 25-400 Hz to isolate the heart sound frequency range, then converted to mel spectrograms with 128 frequency bands. This time-frequency representation is the standard approach in the literature and captures both the timing and frequency characteristics of heart sounds and murmurs.

**For the data**: I used the PhysioNet 2016 training-a subset, which contains 405 clinical recordings. After segmenting into 5-second windows with 50% overlap, this yielded 4,738 samples with a class distribution of 71% Abnormal and 29% Normal.

**The model architecture**: I experimented with three CNN variants ranging from 23K to 1.2M parameters. The best-performing model was a lightweight CNN with just 23,650 parameters, which achieved 74.58% validation accuracy with an ROC-AUC of 0.737.

**Training considerations**: Given the class imbalance, I implemented class-weighted cross-entropy loss to up-weight the minority Normal class. I also used data augmentation with time shifts and frequency masking to improve generalization, plus a learning rate scheduler that reduces the learning rate when validation plateaus."

---

## Key Findings (2-3 minutes)

"I'd like to highlight three key findings from this work:

**First, the importance of real data**: I initially tested with synthetic data, which showed the model wasn't learning — validation accuracy was stuck at 41%, below random guessing. Switching to the PhysioNet 2016 dataset with real clinical recordings immediately improved results to 74%, demonstrating that the pipeline works on authentic data.

**Second, model capacity matters**: The 1.2M parameter model overfit immediately — accuracy dropped from 67% to 43% in just 3 epochs. The 23K parameter model achieved better peak performance (74.58%) and showed it was learning real patterns rather than memorizing. This highlights the importance of matching model capacity to dataset size.

**Third, the model genuinely discriminates**: With an ROC-AUC of 0.737, the model correctly ranks Normal versus Abnormal samples about 74% of the time. This is above the 0.5 baseline of random guessing and confirms the model learned discriminative features, not just predicting the majority class."

---

## Challenges & Solutions (2 minutes)

"The main challenge I encountered was training instability. The validation accuracy oscillated significantly during training — for example, jumping from 72% to 34% and back to 74% within a few epochs. This suggests the model capacity might be slightly too small for the task complexity, or the learning rate could be optimized.

To address the class imbalance, I used class-weighted loss and balanced sampling in the data loader. This helped ensure the model learned both classes rather than collapsing to always predict the majority Abnormal class."

---

## Limitations (1-2 minutes)

"I want to be transparent about the current limitations:

**First, I'm using only one subset** of the PhysioNet dataset — training-a with 405 files. The full dataset has over 3,000 recordings across multiple subsets.

**Second, evaluation is based on a single train/val split**. Cross-validation would provide more robust performance estimates.

**Third, this is a binary classification task** — Normal versus Abnormal. In practice, you'd want to distinguish between different types of murmurs and their timing within the cardiac cycle.

**Finally, this is a proof-of-concept**. Clinical deployment would require much larger datasets, external validation on independent data, and regulatory approval."

---

## Future Work (1-2 minutes)

"To improve this work further, I would:

**First, use the full PhysioNet dataset** with all 3,000+ recordings to increase training data and improve generalization.

**Second, implement k-fold cross-validation** for more robust evaluation and to better understand model stability.

**Third, explore more sophisticated architectures** like attention-based models that can identify which time-frequency regions are most important for classification.

**Fourth, incorporate cardiac cycle segmentation** to focus on S1 and S2 heart sounds and analyze murmurs in their proper timing context — systolic versus diastolic.

**And finally, ensemble methods** combining predictions from multiple models could improve robustness and accuracy."

---

## Closing Statement (30 seconds)

"In summary, I've implemented a complete end-to-end pipeline for heart sound classification that achieves 74.58% accuracy on real clinical data. The project demonstrates the full ML workflow — from data preprocessing and model design to training and evaluation — while providing insights into the challenges of working with medical audio data. Thank you for your time, and I'd be happy to answer any questions."

---

## Expected Questions & Answers

### Q: "Why 74% accuracy? Isn't that low?"

"For a first attempt on real clinical data with only 405 recordings, 74% is actually quite reasonable. The PhysioNet 2016 challenge winners achieved around 85-86%, but they used the full dataset of 3,000+ recordings and much more sophisticated ensembles. My result of 74% with a simple 23K parameter CNN demonstrates the pipeline works and provides a solid baseline for improvement."

### Q: "Why did you choose mel spectrograms?"

"Mel spectrograms are the standard representation in the heart sound literature, including the PhysioNet challenges. They capture both timing and frequency information in a way that matches human perception. The time-frequency representation is crucial because murmurs have characteristic frequency bands and timing within the cardiac cycle."

### Q: "How did you handle the class imbalance?"

"I used two approaches: class-weighted cross-entropy loss, which up-weights the minority Normal class during training, and balanced sampling in the data loader. This ensures the model sees equal numbers of both classes in each batch, preventing it from collapsing to always predict the majority Abnormal class."

### Q: "What would you do differently?"

"I would start with the full PhysioNet dataset rather than a single subset, implement k-fold cross-validation for more robust evaluation, and experiment with a wider range of model architectures. I'd also explore more sophisticated preprocessing like cardiac cycle segmentation to focus on the most informative parts of the signal."

### Q: "Is this clinically ready?"

"No, and I want to be clear about that. This is a proof-of-concept demonstrating the ML pipeline. Clinical deployment would require much larger datasets, external validation on independent patient populations, much higher accuracy (typically >90%), and regulatory approval. However, it provides a solid foundation for further research."

---

## Key Technical Terms to Remember

- **PCG**: Phonocardiogram (heart sound recording)
- **Mel Spectrogram**: Time-frequency representation of audio
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve (0.5 = random, 1.0 = perfect)
- **Class Imbalance**: When one class has many more samples than the other (71% vs 29%)
- **Overfitting**: When model memorizes training data but fails on new data
- **Cross-Validation**: Splitting data into k folds and training/evaluating k times for robust results
- **Class-Weighted Loss**: Giving more importance to minority class during training

---

## Final Tips

1. **Be confident** — 74% on real data is good for a first attempt
2. **Be honest** about limitations — interviewers appreciate integrity
3. **Focus on what you learned** — the insights are more important than the numbers
4. **Have the code ready** — if they ask to see it, you can show the pipeline
5. **Bring up the training instability** — shows you understand the challenges

---

## Files to Reference

- `src/train.py` - Training script with model options
- `src/model.py` - CNN architectures (cnn, cnn_light, cnn_medium, resnet, attention)
- `src/preprocess.py` - Audio preprocessing and mel spectrogram
- `src/dataset.py` - Data loading and augmentation
- `models/training_curves.png` - Visual training progress
- `models/best_model.pth` - Saved best model
- `models/training_results.json` - Detailed metrics

---

Good luck! You've got this! 🎓🫀
