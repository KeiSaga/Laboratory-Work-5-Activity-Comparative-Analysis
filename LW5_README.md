# 🌿 Laboratory Work 5 — Comparative Analysis of Pre-trained CNN Models
### Custom Image Classification Using Transfer Learning

> **Course:** Computer Science — Machine Learning / Deep Learning
> **Dataset:** Tree & Fruit Image Dataset · 20 Classes · 7,188 Images
> **Framework:** TensorFlow / Keras · Google Colab
> **Models Tested:** MobileNetV2 · EfficientNetB0 · ResNet50

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Activity 1 — MobileNetV2](#activity-1--mobilenetv2)
- [Activity 2 — EfficientNetB0](#activity-2--efficientnetb0)
- [Activity 3 — ResNet50](#activity-3--resnet50)
- [Performance Comparison Table](#performance-comparison-table)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)

---

## Overview

This laboratory work applies **transfer learning** using three pre-trained CNN architectures to classify 20 species of trees and fruits. Each model was initialized with **ImageNet weights**, frozen during feature extraction, and trained with a custom classification head on the dataset. Models were evaluated using Accuracy, Loss, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and AUC Score.

---

## Dataset

| Property | Value |
|---|---|
| Total Images | 7,188 |
| Number of Classes | 20 |
| Training Set | 5,751 images (80%) |
| Validation Set | 1,437 images (20%) |
| Input Size | 224 × 224 px (required by pre-trained models) |
| Batch Size | 32 |

**20 Classes:**
`ARATILES` · `AVOCADO` · `Baobab (Adansonia digitata)` · `Breadfruit Tree (Artocarpus altilis)` · `COCOA` · `Cedar of Lebanon (Cedrus libani)` · `DUHAT` · `Dragon Blood Tree` · `Eucalyptus` · `GUAVA` · `KAMIAS` · `Kapur Tree (Dryobalanops aromatica)` · `LANGKA` · `LANZONES` · `MANGOSTEEN` · `Monkey Pod Tree (Samanea saman)` · `PAPAYA` · `RAMBUTAN` · `Soursop Tree (Annona muricata)` · `Wollemi Pine (Wollemia nobilis)`

---

## Model Architecture

All three models share the same transfer learning wrapper:

```
Input (224 × 224 × 3)
  → Rescaling (÷ 255)
  → Pre-trained Base (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(20)  ← output logits
```

| Component | Detail |
|---|---|
| Base weights | ImageNet (frozen) |
| Optimizer | Adam (lr = 0.0001) |
| Loss | SparseCategoricalCrossentropy (from_logits=True) |
| Max Epochs | 10 |
| Callback | EarlyStopping — monitors val_loss, patience=3, restores best weights |

---

## Training Results

### MobileNetV2 — Training Progress (All 10 Epochs)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 15.61% | 2.8706 | 39.94% | 2.3711 |
| 2 | 35.39% | 2.2426 | 58.73% | 1.8409 |
| 3 | 49.35% | 1.8104 | 68.89% | 1.4696 |
| 7 | 76.37% | 0.9483 | 87.06% | 0.7285 |
| 8 | 79.62% | 0.8246 | 90.12% | 0.6324 |
| 9 | 82.68% | 0.7316 | 91.86% | 0.5463 |
| **10** | **85.72%** | **0.6435** | **92.55%** | **0.4692** |

### EfficientNetB0 — Training (Early Stop at Epoch 3)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 4.33% | 3.0080 | 5.43% | 2.9909 |
| 2 | 5.13% | 2.9956 | 5.22% | 2.9904 |
| 3 | 5.04% | 2.9940 | 5.22% | 2.9912 |

> ⚠️ Early Stopping triggered at Epoch 3 — model failed to learn (stuck near random baseline for 20 classes)

### ResNet50 — Training (Early Stop at Epoch 3)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 5.39% | 3.1021 | 6.89% | 2.9755 |
| 2 | 6.99% | 2.9776 | 7.93% | 2.9597 |
| 3 | 7.62% | 2.9638 | 9.39% | 2.9476 |

> ⚠️ Early Stopping triggered at Epoch 3 — minimal improvement, stuck near random baseline

---

## Activity 1 — MobileNetV2

> **Result: ✅ Best Performing Model**

### Screenshots

> **Training Curves — Accuracy & Loss**
> *(Add screenshot: `screenshots/MobileNetV2_training_curves.png`)*

> **Confusion Matrix**
> *(Add screenshot: `screenshots/MobileNetV2_confusion_matrix.png`)*

> **ROC Curve**
> *(Add screenshot: `screenshots/MobileNetV2_roc_curve.png`)*

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ARATILES | 0.96 | 0.97 | 0.96 | 67 |
| AVOCADO | 0.91 | 0.95 | 0.93 | 76 |
| Baobab (Adansonia digitata) | 0.93 | 0.89 | 0.91 | 61 |
| Breadfruit Tree (Artocarpus altilis) | 0.79 | 0.91 | 0.84 | 53 |
| COCOA | 1.00 | 0.91 | 0.95 | 81 |
| Cedar of Lebanon (Cedrus libani) | 0.90 | 0.96 | 0.92 | 90 |
| DUHAT | 0.94 | 0.78 | 0.85 | 82 |
| Dragon Blood Tree | 0.90 | 0.98 | 0.94 | 47 |
| Eucalyptus | 0.91 | 0.94 | 0.92 | 78 |
| GUAVA | 0.90 | 0.91 | 0.91 | 82 |
| KAMIAS | 0.88 | 0.96 | 0.92 | 78 |
| Kapur Tree (Dryobalanops aromatica) | 0.92 | 0.93 | 0.92 | 86 |
| LANGKA | 0.98 | 0.88 | 0.93 | 73 |
| LANZONES | 0.94 | 0.95 | 0.95 | 66 |
| MANGOSTEEN | 0.88 | 0.95 | 0.91 | 75 |
| Monkey Pod Tree (Samanea saman) | 0.92 | 0.95 | 0.93 | 57 |
| PAPAYA | 1.00 | 0.96 | **0.98** ⭐ | 70 |
| RAMBUTAN | 0.95 | 0.96 | 0.95 | 73 |
| Soursop Tree (Annona muricata) | 0.96 | 0.88 | 0.92 | 80 |
| Wollemi Pine (Wollemia nobilis) | 0.97 | 0.95 | 0.96 | 62 |
| **Macro Avg** | **0.93** | **0.93** | **0.93** | 1437 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | 1437 |

**Overall Validation Accuracy:** `92.55%` &nbsp;·&nbsp; **AUC Score:** `0.9909`

> **Best class:** PAPAYA (F1 = 0.98) · **Weakest class:** Breadfruit Tree (F1 = 0.84) · **Lowest recall:** DUHAT (0.78)

---

## Activity 2 — EfficientNetB0

> **Result: ❌ Failed to Converge — Early Stopping at Epoch 3**

### Screenshots

> **Training Curves — Accuracy & Loss**
> *(Add screenshot: `screenshots/EfficientNetB0_training_curves.png`)*

> **Confusion Matrix**
> *(Add screenshot: `screenshots/EfficientNetB0_confusion_matrix.png`)*

> **ROC Curve**
> *(Add screenshot: `screenshots/EfficientNetB0_roc_curve.png`)*

### Classification Report Summary

The model predicted almost exclusively the **KAMIAS** class for all 1,437 validation images — a classic **class collapse** failure. All other 19 classes received Precision = 0.00 and Recall = 0.00.

| Metric | Value |
|---|---|
| Validation Accuracy | 5.43% |
| AUC Score | 0.5280 (≈ random) |
| Macro Precision | 0.0027 |
| Macro Recall | 0.0500 |
| Macro F1-Score | 0.0051 |

### Root Cause

EfficientNetB0 includes **built-in input preprocessing** internally. Adding `Rescaling(1./255)` on top caused a double-normalization conflict — pixel values that should be in [0, 255] were rescaled to [0, 1] before being passed to a model expecting [0, 255], breaking the internal batch normalization layers and preventing any learning.

**Fix for next run:**
```python
# Remove Rescaling layer when using EfficientNetB0
# Use model-specific preprocessing instead:
from tensorflow.keras.applications.efficientnet import preprocess_input
layers.Lambda(lambda x: preprocess_input(x))
```

---

## Activity 3 — ResNet50

> **Result: ❌ Failed to Converge — Early Stopping at Epoch 3**

### Screenshots

> **Training Curves — Accuracy & Loss**
> *(Add screenshot: `screenshots/ResNet50_training_curves.png`)*

> **Confusion Matrix**
> *(Add screenshot: `screenshots/ResNet50_confusion_matrix.png`)*

> **ROC Curve**
> *(Add screenshot: `screenshots/ResNet50_roc_curve.png`)*

### Classification Report Summary

ResNet50 showed near-random performance across all 20 classes, concentrating predictions on the **Baobab** class (recall 0.67). All other classes were essentially ignored.

| Metric | Value |
|---|---|
| Validation Accuracy | 6.89% |
| AUC Score | 0.5900 (≈ random) |
| Macro Precision | 0.0857 |
| Macro Recall | 0.0719 |
| Macro F1-Score | 0.0414 |

### Root Cause

Same double-normalization conflict as EfficientNetB0. ResNet50 also has an internal `preprocess_input` that expects raw [0, 255] pixel values.

**Fix for next run:**
```python
from tensorflow.keras.applications.resnet50 import preprocess_input
layers.Lambda(lambda x: preprocess_input(x))
```

---

## Performance Comparison Table

| Model | Train Acc | Train Loss | Val Acc | Val Loss | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|---|---|---|
| **MobileNetV2** ✅ | 85.72% | 0.6435 | **92.55%** | **0.4692** | **0.9265** | **0.9274** | **0.9256** | **0.9909** |
| EfficientNetB0 ❌ | 5.04% | 2.9940 | 5.43% | 2.9912 | 0.0027 | 0.0500 | 0.0051 | 0.5280 |
| ResNet50 ❌ | 7.62% | 2.9638 | 6.89% | 2.9476 | 0.0857 | 0.0719 | 0.0414 | 0.5900 |
| Teachable Machine | — | — | — | — | — | — | — | — |
| LW4 Baseline (Custom CNN) | — | — | 99.49% | 0.015 | 0.84 | 0.81 | 0.81 | 0.9627 |
| LW4 Improved (Enhanced CNN) | — | — | — | — | — | — | — | — |
| Best / Final Model | — | — | — | — | — | — | — | — |

> 🏆 **Winner among LW5 models: MobileNetV2** — only model that successfully converged, with 92.55% val accuracy and AUC of 0.9909.

---

## Grad-CAM Explainability

Grad-CAM was applied to all 3 trained models on the same test image to visualize which regions influenced each model's prediction.

| Model | Last Conv Layer Used | Observation |
|---|---|---|
| MobileNetV2 | `Conv_1` | ✅ Focused activation on the target object |
| EfficientNetB0 | `top_conv` | ⚠️ Diffuse/scattered — model did not learn discriminative features |
| ResNet50 | `conv5_block3_out` | ⚠️ Diffuse/scattered — model did not learn discriminative features |

> MobileNetV2 produced the most meaningful heatmap, consistent with it being the only model that converged.

### Screenshots

> **Grad-CAM Side-by-Side Comparison (All 3 Models)**
> *(Add screenshot: `screenshots/gradcam_comparison.png`)*

---

## Key Findings

**1. MobileNetV2 was the only successful model** — achieved 92.55% validation accuracy and AUC 0.9909, with strong F1-scores across all 20 classes (macro avg 0.93).

**2. EfficientNetB0 and ResNet50 both failed due to an input preprocessing conflict.** Both architectures include built-in normalization and expect raw [0–255] pixel values. The extra `Rescaling(1./255)` layer caused a distribution mismatch that broke internal batch normalization, preventing convergence entirely.

**3. MobileNetV2 is compatible with external rescaling** — it does not include built-in preprocessing, making it the safest choice when using `Rescaling(1./255)` in the model pipeline.

**4. MobileNetV2 vs LW4 Custom CNN** — The LW4 custom CNN reported 99.49% accuracy but had a lower AUC (0.9627) vs MobileNetV2's 0.9909. MobileNetV2 demonstrates stronger class discrimination despite lower raw accuracy, suggesting the LW4 model may have been overfitting on the validation set.

**5. Recommended fix** for EfficientNetB0 and ResNet50 — replace the generic `Rescaling` layer with model-specific `preprocess_input` functions to allow these models to converge properly on a re-run.

---

## Repository Structure

```
LW5/
├── LW5.ipynb                               # Main Colab notebook
├── README.md                               # This file
└── screenshots/
    ├── MobileNetV2_training_curves.png     # Activity 1
    ├── MobileNetV2_confusion_matrix.png    # Activity 1
    ├── MobileNetV2_roc_curve.png           # Activity 1
    ├── EfficientNetB0_training_curves.png  # Activity 2
    ├── EfficientNetB0_confusion_matrix.png # Activity 2
    ├── EfficientNetB0_roc_curve.png        # Activity 2
    ├── ResNet50_training_curves.png        # Activity 3
    ├── ResNet50_confusion_matrix.png       # Activity 3
    ├── ResNet50_roc_curve.png              # Activity 3
    └── gradcam_comparison.png             # All 3 models side-by-side
```

---

## Dependencies

```
tensorflow >= 2.x
scikit-learn
matplotlib
numpy
opencv-python (cv2)
Pillow
```

---

*Laboratory Work 5 · Comparative Analysis of Pre-trained CNN Models · Transfer Learning*
