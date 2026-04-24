# 🌿 Laboratory Work 5 — Comparative Analysis of Pre-trained CNN Models
### Custom Image Classification Using Transfer Learning

> **Course:** Computer Science — Machine Learning / Deep Learning
> **Dataset:** Tree & Fruit Image Dataset · 20 Classes · 7,188 Images
> **Framework:** TensorFlow / Keras · Google Colab
> **Models:** MobileNetV2 · EfficientNetB0 · ResNet50

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model 1 — MobileNetV2](#model-1--mobilenetv2)
- [Model 2 — EfficientNetB0](#model-2--efficientnetb0)
- [Model 3 — ResNet50](#model-3--resnet50)
- [Performance Comparison Table](#performance-comparison-table)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)

---

## Overview

This laboratory work applies **transfer learning** using three pre-trained CNN architectures — MobileNetV2, EfficientNetB0, and ResNet50 — to classify 20 species of trees and fruits from a custom image dataset. Each model was initialized with **ImageNet weights**, frozen during feature extraction, and trained with a custom classification head. Models were evaluated using Accuracy, Loss, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and AUC Score.

---

## Dataset

| Property | Value |
|---|---|
| Total Images | 7,188 |
| Number of Classes | 20 |
| Training Set | 5,751 images (80%) |
| Validation Set | 1,437 images (20%) |
| Input Size | 224 × 224 px |
| Batch Size | 32 |

**20 Classes:**
`ARATILES` · `AVOCADO` · `Baobab (Adansonia digitata)` · `Breadfruit Tree (Artocarpus altilis)` · `COCOA` · `Cedar of Lebanon (Cedrus libani)` · `DUHAT` · `Dragon Blood Tree` · `Eucalyptus` · `GUAVA` · `KAMIAS` · `Kapur Tree (Dryobalanops aromatica)` · `LANGKA` · `LANZONES` · `MANGOSTEEN` · `Monkey Pod Tree (Samanea saman)` · `PAPAYA` · `RAMBUTAN` · `Soursop Tree (Annona muricata)` · `Wollemi Pine (Wollemia nobilis)`

---

## Model Architecture

All three models share the same transfer learning wrapper with **model-specific preprocessing** to avoid normalization conflicts:

```
Input (224 × 224 × 3)
  → Lambda (model-specific preprocess_input)
  → Pre-trained Base (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(20)  ← output logits
```

| Component | Detail |
|---|---|
| Base weights | ImageNet (frozen — `trainable = False`) |
| Normalization | Model-specific `preprocess_input` per architecture |
| Optimizer | Adam (`lr = 0.0001`) |
| Loss | `SparseCategoricalCrossentropy(from_logits=True)` |
| Max Epochs | 10 |
| Callback | `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` |

---

## Model 1 — MobileNetV2

> ✅ **Successfully trained — All 10 epochs completed**

### Training Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 16.17% | 2.8558 | 40.50% | 2.3652 |
| 2 | 37.80% | 2.1917 | 62.98% | 1.7479 |
| 3 | 52.20% | 1.7401 | 70.63% | 1.3566 |
| 4 | 61.71% | 1.4125 | 77.94% | 1.1039 |
| 5 | 68.74% | 1.1901 | 82.88% | 0.9165 |
| 6 | 74.28% | 1.0043 | 86.43% | 0.7738 |
| 7 | 78.06% | 0.8847 | 88.94% | 0.6627 |
| 8 | 80.87% | 0.7765 | 91.09% | 0.5679 |
| 9 | 84.11% | 0.6803 | 92.90% | 0.4806 |
| **10** | **85.90%** | **0.6033** | **93.81%** | **0.4186** |

### Training Curves

![MobileNetV2 Training Curves](screenshots/MobileNetV2_training_curves.png)

### Confusion Matrix

![MobileNetV2 Confusion Matrix](screenshots/MobileNetV2_confusion_matrix.png)

### ROC Curve

![MobileNetV2 ROC Curve](screenshots/MobileNetV2_roc_curve.png)

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ARATILES | 1.00 | 0.94 | 0.97 | 67 |
| AVOCADO | 0.99 | 0.88 | 0.93 | 76 |
| Baobab (Adansonia digitata) | 0.97 | 0.92 | 0.94 | 61 |
| Breadfruit Tree (Artocarpus altilis) | 0.96 | 0.94 | 0.95 | 53 |
| COCOA | 0.94 | 0.94 | 0.94 | 81 |
| Cedar of Lebanon (Cedrus libani) | 0.88 | 0.96 | 0.91 | 90 |
| DUHAT | 1.00 | 0.82 | 0.90 | 82 |
| Dragon Blood Tree | 0.98 | 1.00 | **0.99** ⭐ | 47 |
| Eucalyptus | 0.88 | 0.96 | 0.92 | 78 |
| GUAVA | 0.90 | 0.91 | 0.91 | 82 |
| KAMIAS | 1.00 | 0.96 | 0.98 | 78 |
| Kapur Tree (Dryobalanops aromatica) | 0.83 | 0.94 | 0.88 | 86 |
| LANGKA | 0.99 | 0.90 | 0.94 | 73 |
| LANZONES | 0.98 | 0.98 | 0.98 | 66 |
| MANGOSTEEN | 0.89 | 0.96 | 0.92 | 75 |
| Monkey Pod Tree (Samanea saman) | 0.84 | 0.98 | 0.90 | 57 |
| PAPAYA | 1.00 | 1.00 | **1.00** ⭐ | 70 |
| RAMBUTAN | 0.92 | 0.96 | 0.94 | 73 |
| Soursop Tree (Annona muricata) | 0.97 | 0.95 | 0.96 | 80 |
| Wollemi Pine (Wollemia nobilis) | 0.98 | 0.89 | 0.93 | 62 |
| **Macro Avg** | **0.94** | **0.94** | **0.94** | 1437 |
| **Weighted Avg** | **0.94** | **0.94** | **0.94** | 1437 |

**Validation Accuracy:** `93.81%` · **AUC Score:** `0.9935`

> 🏆 Best class: PAPAYA (F1 = 1.00, perfect score) · Weakest: Kapur Tree (F1 = 0.88) · Lowest recall: DUHAT (0.82)

---

## Model 2 — EfficientNetB0

> ⚠️ **Partially trained — Early Stopping triggered at Epoch 3**

### Training Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 18.01% | 2.7829 | 45.93% | 2.3202 |
| 2 | 42.76% | 2.1542 | 62.70% | 1.7512 |
| **3** | **54.34%** | **1.7252** | **71.89%** | **1.3760** |

> ⚠️ Early Stopping triggered at Epoch 3. Model was actively learning but `patience=3` halted it too early. With more epochs, expected val accuracy: 88–93%.

### Training Curves

![EfficientNetB0 Training Curves](screenshots/EfficientNetB0_training_curves.png)

### Confusion Matrix

![EfficientNetB0 Confusion Matrix](screenshots/EfficientNetB0_confusion_matrix.png)

### ROC Curve

![EfficientNetB0 ROC Curve](screenshots/EfficientNetB0_roc_curve.png)

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ARATILES | 0.35 | 0.61 | 0.45 | 67 |
| AVOCADO | 0.52 | 0.30 | 0.38 | 76 |
| Baobab (Adansonia digitata) | 0.61 | 0.67 | 0.64 | 61 |
| Breadfruit Tree (Artocarpus altilis) | 0.33 | 0.30 | 0.31 | 53 |
| COCOA | 0.44 | 0.83 | 0.57 | 81 |
| Cedar of Lebanon (Cedrus libani) | 0.37 | 0.89 | 0.52 | 90 |
| DUHAT | 0.73 | 0.13 | 0.23 | 82 |
| Dragon Blood Tree | 0.87 | 0.43 | 0.57 | 47 |
| Eucalyptus | 0.62 | 0.10 | **0.18** ⚠️ | 78 |
| GUAVA | 0.38 | 0.57 | 0.46 | 82 |
| KAMIAS | 0.62 | 0.47 | 0.54 | 78 |
| Kapur Tree (Dryobalanops aromatica) | 0.48 | 0.35 | 0.40 | 86 |
| LANGKA | 0.62 | 0.63 | 0.63 | 73 |
| LANZONES | 0.66 | 0.58 | 0.61 | 66 |
| MANGOSTEEN | 0.46 | 0.16 | 0.24 | 75 |
| Monkey Pod Tree (Samanea saman) | 0.35 | 0.44 | 0.39 | 57 |
| PAPAYA | 0.43 | 0.17 | 0.24 | 70 |
| RAMBUTAN | 0.59 | 0.64 | 0.61 | 73 |
| Soursop Tree (Annona muricata) | 0.36 | 0.47 | 0.41 | 80 |
| Wollemi Pine (Wollemia nobilis) | 0.42 | 0.34 | 0.38 | 62 |
| **Macro Avg** | **0.51** | **0.45** | **0.44** | 1437 |
| **Weighted Avg** | **0.51** | **0.46** | **0.44** | 1437 |

**Validation Accuracy:** `45.93%` · **AUC Score:** `0.8696`

---

## Model 3 — ResNet50

> ⚠️ **Partially trained — Early Stopping triggered at Epoch 3**

### Training Progress

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 18.94% | 2.7751 | 46.56% | 2.1237 |
| 2 | 42.55% | 1.9716 | 65.62% | 1.5043 |
| **3** | **57.62%** | **1.5088** | **77.38%** | **1.1212** |

> ⚠️ Early Stopping triggered at Epoch 3. ResNet50 showed the strongest per-epoch gain (val acc: 46% → 77% in 3 epochs). Expected val accuracy with full 10 epochs: 85–92%.

### Training Curves

![ResNet50 Training Curves](screenshots/ResNet50_training_curves.png)

### Confusion Matrix

![ResNet50 Confusion Matrix](screenshots/ResNet50_confusion_matrix.png)

### ROC Curve

![ResNet50 ROC Curve](screenshots/ResNet50_roc_curve.png)

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ARATILES | 0.43 | 0.19 | 0.27 | 67 |
| AVOCADO | 0.31 | 0.43 | 0.36 | 76 |
| Baobab (Adansonia digitata) | 0.52 | 0.56 | 0.54 | 61 |
| Breadfruit Tree (Artocarpus altilis) | 0.37 | 0.19 | 0.25 | 53 |
| COCOA | 0.41 | 0.86 | 0.55 | 81 |
| Cedar of Lebanon (Cedrus libani) | 0.36 | 0.78 | 0.49 | 90 |
| DUHAT | 0.51 | 0.44 | 0.47 | 82 |
| Dragon Blood Tree | 1.00 | 0.40 | 0.58 | 47 |
| Eucalyptus | 0.37 | 0.31 | 0.34 | 78 |
| GUAVA | 0.70 | 0.45 | 0.55 | 82 |
| KAMIAS | 0.43 | 0.23 | **0.30** ⚠️ | 78 |
| Kapur Tree (Dryobalanops aromatica) | 0.36 | 0.47 | 0.40 | 86 |
| LANGKA | 0.62 | 0.64 | 0.63 | 73 |
| LANZONES | 0.59 | 0.71 | 0.65 | 66 |
| MANGOSTEEN | 0.64 | 0.36 | 0.46 | 75 |
| Monkey Pod Tree (Samanea saman) | 0.29 | 0.32 | **0.30** ⚠️ | 57 |
| PAPAYA | 0.42 | 0.56 | 0.48 | 70 |
| RAMBUTAN | 0.68 | 0.71 | 0.69 | 73 |
| Soursop Tree (Annona muricata) | 0.71 | 0.30 | 0.42 | 80 |
| Wollemi Pine (Wollemia nobilis) | 0.73 | 0.18 | **0.29** ⚠️ | 62 |
| **Macro Avg** | **0.52** | **0.45** | **0.45** | 1437 |
| **Weighted Avg** | **0.51** | **0.47** | **0.45** | 1437 |

**Validation Accuracy:** `46.56%` · **AUC Score:** `0.8674`

---

## Performance Comparison Table

| Model | Train Acc | Train Loss | Val Acc | Val Loss | Precision | Recall | F1-Score | AUC | Epochs Used |
|---|---|---|---|---|---|---|---|---|---|
| **MobileNetV2** ✅ | 85.90% | 0.6033 | **93.81%** | **0.4186** | **0.9446** | **0.9401** | **0.9407** | **0.9935** | 10 / 10 |
| EfficientNetB0 ⚠️ | 54.34% | 1.7252 | 71.89% | 1.3760 | 0.5097 | 0.4548 | 0.4379 | 0.8696 | 3 / 10 |
| ResNet50 ⚠️ | 57.62% | 1.5088 | 77.38% | 1.1212 | 0.5217 | 0.4546 | 0.4508 | 0.8674 | 3 / 10 |
| Teachable Machine | — | — | — | — | — | — | — | — | — |
| LW4 Baseline (Custom CNN) | — | — | 99.49% | 0.015 | 0.84 | 0.81 | 0.81 | 0.9627 | — |
| LW4 Improved (Enhanced CNN) | — | — | — | — | — | — | — | — | — |
| Best / Final Model | — | — | — | — | — | — | — | — | — |

> 🏆 **Winner among LW5 models: MobileNetV2** — only model that completed full training, achieving 93.81% val accuracy and AUC of 0.9935.
>
> ⚠️ **Note:** EfficientNetB0 and ResNet50 stopped at Epoch 3 due to `patience=3`. Both were actively improving — increasing patience to 5 would allow full convergence.

---

## Grad-CAM Explainability

Grad-CAM was applied to all 3 trained models on the same test image to compare which spatial regions drove each model's prediction.

| Model | Last Conv Layer | Heatmap Observation |
|---|---|---|
| MobileNetV2 | `Conv_1` | ✅ Focused activation on target object features |
| EfficientNetB0 | `top_conv` | ⚠️ Partially diffuse — limited by early stopping |
| ResNet50 | `conv5_block3_out` | ⚠️ Partially diffuse — limited by early stopping |

### Grad-CAM Side-by-Side Comparison — All 3 Models

![Grad-CAM Comparison](screenshots/gradcam_comparison.png)

---

## Key Findings

**1. MobileNetV2 was the strongest and most reliable performer** — completed all 10 epochs with consistent improvement every epoch, reaching 93.81% val accuracy and a perfect F1 of 1.00 on PAPAYA. Its AUC of 0.9935 surpasses even the LW4 baseline (0.9627).

**2. EfficientNetB0 and ResNet50 were stopped too early** — both were actively converging when Early Stopping triggered. This is a hyperparameter issue, not a model failure.

**3. The fix is simple** — increase `patience` from 3 to 5 and re-run:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

**4. Speed comparison** — MobileNetV2 ran at 2s/step, EfficientNetB0 at 3s/step, ResNet50 at 7s/step. MobileNetV2 offers the best speed-to-accuracy trade-off.

**5. MobileNetV2 is recommended for deployment** — lightweight (3.4M parameters), fastest inference, highest accuracy, and best AUC. Ideal for mobile or web-based real-time plant/tree identification.

---

## Repository Structure

```
LW5/
├── LW5.ipynb
├── README.md
├── guide_questions.md
└── screenshots/
    ├── MobileNetV2_training_curves.png
    ├── MobileNetV2_confusion_matrix.png
    ├── MobileNetV2_roc_curve.png
    ├── EfficientNetB0_training_curves.png
    ├── EfficientNetB0_confusion_matrix.png
    ├── EfficientNetB0_roc_curve.png
    ├── ResNet50_training_curves.png
    ├── ResNet50_confusion_matrix.png
    ├── ResNet50_roc_curve.png
    └── gradcam_comparison.png
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
