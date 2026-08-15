# NeuroGuard AI — Brain Tumor MRI Classification (Baseline Model)

Deep learning system for classifying brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, No Tumor**. This repo contains the baseline model (EfficientNetB1) — part of a larger project comparing baseline vs. GAN-enhanced training data.

## Overview

- **Framework:** PyTorch
- **Architecture:** EfficientNetB1 (ImageNet-pretrained) + custom classifier head
- **Training:** Staged fine-tuning (frozen backbone → full unfreeze), 5-Fold Stratified Cross-Validation, 100-epoch final model
- **Input:** 224×224 grayscale MRI images (converted to RGB)
- **Dataset:** 5,299 training images, 1,600 test images (400/class, balanced)

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 94.19% |
| Balanced Accuracy | 94.19% |
| Macro F1 | 0.9414 |
| Cohen's Kappa | 0.9225 |
| MCC | 0.9241 |
| Macro ROC-AUC | 0.9738 |
| 5-Fold CV Mean Accuracy | 98.92% ± 0.40% |

**Class-wise F1:** Pituitary 0.996 · No Tumor 0.936 · Meningioma 0.926 · Glioma 0.907

**Known limitation:** Glioma has the lowest recall (83.25%) with confusion mainly toward Meningioma and No Tumor. 48 tumor cases were misclassified as "No Tumor" — the most clinically significant error type, flagged for improvement in the next (GAN-enhanced) phase.

## Repository Contents

```
evaluation_results/
├── confusion_matrix.png / confusion_matrix_normalized.png
├── roc_curve.png / precision_recall_curve.png
├── calibration_curve.png
├── class_metrics.csv / overall_metrics.csv / prediction_results.csv
├── gradcam/ / focused_gradcam/
├── correct_predictions/ / misclassified_predictions/
└── final_report.txt
```

## Evaluation Pipeline

Full research-grade evaluation including: per-class precision/recall/specificity/F1, confusion matrices (raw + normalized), ROC & Precision-Recall curves (one-vs-rest), confidence calibration (ECE, Brier score), Grad-CAM++ explainability (with focused activation regions and top-3 activation zones), and false-positive/false-negative analysis with special focus on tumor-missed-as-No-Tumor cases.

## Next Steps

Building a Conditional GAN to generate synthetic training images, balancing the dataset, and retraining the same architecture on real + synthetic data — to measure whether synthetic data augmentation improves classification performance (particularly for Glioma).

## Disclaimer

This is an AI-assisted screening tool for research/educational purposes and does not replace professional medical diagnosis. Final decisions should always be made by a qualified medical professional.
