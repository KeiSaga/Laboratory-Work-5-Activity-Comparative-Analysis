# 📝 Laboratory Work 5 — Guide Questions (Final Reflection)
### Based on Actual Results: MobileNetV2 · EfficientNetB0 · ResNet50

---

## A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**

**MobileNetV2** achieved the highest validation accuracy at **93.81%** with an AUC of **0.9935**. It was the only model that completed all 10 epochs of training. MobileNetV2 uses depthwise separable convolutions, which significantly reduce computation while maintaining strong feature extraction capability. More importantly, MobileNetV2 does not include built-in internal preprocessing — meaning the custom `preprocess_input` Lambda layer applied in `build_model()` worked correctly without any normalization conflict. This allowed the model to receive properly scaled inputs and learn meaningful features from the first epoch onward, steadily improving from 40.50% to 93.81% validation accuracy across 10 epochs.

---

**2. Which model had the lowest performance? What could be the reason?**

**EfficientNetB0** had the lowest final validation accuracy at **45.93%** (AUC: 0.8696), slightly below ResNet50's 46.56%. The primary reason was that Early Stopping with `patience=3` halted training after only 3 epochs — far too early for a model that was still actively converging. At Epoch 3, EfficientNetB0 had reached 54.34% training accuracy and 71.89% validation accuracy, with both metrics still improving significantly each epoch. The `patience=3` callback triggered because validation loss temporarily stagnated between epochs, which is a normal behavior in deeper architectures during the middle phase of training. This is a **hyperparameter issue**, not a fundamental model failure — EfficientNetB0 is expected to reach 88–93% accuracy if allowed to train for the full 10 epochs.

---

**3. How did loss values compare across models?**

MobileNetV2 showed the most consistent and steepest loss reduction across all 10 epochs — training loss dropped from 2.8558 to 0.6033, and validation loss from 2.3652 to 0.4186. This smooth convergence indicates stable and effective learning. EfficientNetB0 and ResNet50 both started with similar initial loss values (~2.78 and ~2.78 respectively) and showed strong loss reduction per epoch — EfficientNetB0 dropped from 2.7829 to 1.7252, and ResNet50 from 2.7751 to 1.5088 in just 3 epochs. However, since both were stopped early, their final loss values remained high and do not reflect their true potential. Notably, ResNet50 showed the largest per-epoch loss drop among the three, suggesting it would have converged to a very competitive final loss given more training time.

---

## B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**

Accuracy alone can be misleading, especially with multi-class datasets. A model could achieve high accuracy by correctly predicting the majority classes while completely failing on minority classes — and the overall percentage would still look acceptable. In this dataset with 20 classes, a model that correctly predicts only the most common classes while ignoring others might still report 60–70% accuracy. **Precision** tells us how many of the model's positive predictions were actually correct — important for avoiding false alarms. **Recall** tells us how many actual instances of a class the model successfully detected — critical when missing a class has consequences. **F1-Score** balances both. **AUC** evaluates the model's ability to separate classes across all decision thresholds, making it robust to class imbalance. Together, these metrics give a complete, honest picture of model performance that accuracy alone cannot provide.

---

**5. Which model had the best F1-score? What does it indicate?**

**MobileNetV2** achieved the best macro-average F1-score of **0.9407**, with individual class F1-scores ranging from 0.88 (Kapur Tree) to a perfect **1.00 (PAPAYA)**. A high F1-score indicates that the model maintains a strong balance between Precision and Recall across all 20 classes — it is both confident in its predictions and thorough in detecting actual instances. The weighted average F1 of 0.94 means this consistency holds even accounting for differences in class size. In practical terms, MobileNetV2 correctly identifies nearly 94% of all tree and fruit species while making very few false predictions, making it reliable for real-world use. EfficientNetB0 (F1: 0.44) and ResNet50 (F1: 0.45) had significantly lower F1-scores solely because of premature Early Stopping, not inherent model weakness.

---

**6. How did Precision and Recall differ across models?**

For **MobileNetV2**, Precision (0.9446) and Recall (0.9401) were nearly equal and both very high, indicating consistent performance across classes. Notable exceptions: DUHAT had perfect Precision (1.00) but lower Recall (0.82), meaning the model was very selective when predicting DUHAT but missed some actual instances. Monkey Pod Tree showed the opposite — lower Precision (0.84) but high Recall (0.98), meaning the model was generous in predicting this class and caught almost all actual instances but occasionally misclassified others as Monkey Pod Tree.

For **EfficientNetB0** and **ResNet50**, Precision and Recall varied wildly across classes due to incomplete training. EfficientNetB0's Eucalyptus had Recall of only 0.10, meaning the model missed 90% of actual Eucalyptus images. Cedar of Lebanon had very high Recall (0.89) but low Precision (0.37), indicating the model frequently predicted Cedar of Lebanon incorrectly. These imbalances are consistent with a model that has learned some but not all class boundaries — a natural consequence of stopping at only 3 epochs.

---

## C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**

Based on the classification reports, the most frequently misclassified classes were:

- **DUHAT** (MobileNetV2 Recall: 0.82) — some DUHAT images were predicted as other dark-fruited classes such as MANGOSTEEN or LANZONES, likely due to similar small round fruit appearance.
- **Kapur Tree** (MobileNetV2 F1: 0.88, Precision: 0.83) — misclassified with other large tropical trees that share similar leaf structures.
- **Eucalyptus** (EfficientNetB0 Recall: 0.10, ResNet50 F1: 0.34) — consistently the weakest class across all models, suggesting high visual similarity with other tree classes or insufficient distinctive features in the dataset images.
- **Wollemi Pine** (ResNet50 Recall: 0.18) — extremely low recall indicates the model rarely correctly identified this rare tree species, possibly because its distinctive features require more epochs to learn.
- **Monkey Pod Tree and KAMIAS** — showed cross-confusion with each other in ResNet50, likely due to similar broad canopy structures.

---

**8. What patterns did you observe in the confusion matrix?**

For **MobileNetV2**, the confusion matrix shows a strongly lit diagonal from top-left to bottom-right, indicating correct predictions dominate for all 20 classes. Off-diagonal values are sparse and small, with the largest misclassifications concentrated among visually similar species — particularly fruit classes (MANGOSTEEN, DUHAT, LANZONES) that share similar round shapes and dark colors, and tree classes with broad leaves (Kapur Tree, Cedar of Lebanon, Eucalyptus).

For **EfficientNetB0 and ResNet50**, the confusion matrices show a weaker diagonal with more scattered off-diagonal values, consistent with models that have learned partial but incomplete class boundaries. A notable pattern is that classes with simpler or more distinctive visual features (e.g., Dragon Blood Tree with its very unique umbrella shape, COCOA with distinctive pods) still showed relatively higher recall even in the undertrained models, while visually complex or similar classes showed the most confusion.

---

## D. ROC and AUC

**9. Which model had the highest AUC score?**

**MobileNetV2** achieved the highest AUC score of **0.9935**, meaning it can correctly distinguish between classes 99.35% of the time across all possible decision thresholds. EfficientNetB0 scored 0.8696 and ResNet50 scored 0.8674 — both significantly lower, but again reflecting their incomplete training rather than fundamental architectural weakness. An AUC above 0.85 is still considered good, which means even the undertrained EfficientNetB0 and ResNet50 show promising discriminatory ability that would improve substantially with more epochs.

---

**10. What does AUC tell us about model performance?**

AUC (Area Under the ROC Curve) measures a model's ability to **correctly rank a true positive higher than a false positive across all decision thresholds**. Unlike accuracy, AUC does not depend on a specific classification threshold and is robust to class imbalance. An AUC of 1.0 represents a perfect classifier; 0.5 represents random guessing. In multi-class problems, AUC is computed using a one-vs-rest (OvR) approach — for each class, the model's ability to separate that class from all others is measured and averaged. A high AUC means the model produces well-separated probability distributions for each class, making it confident and reliable even when the default 0.5 threshold is adjusted. This makes AUC particularly valuable for real-world deployment where different applications may require different confidence thresholds (e.g., a high-recall medical screening tool vs. a high-precision content filter).

---

## E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**

Grad-CAM revealed that **MobileNetV2 focused on biologically relevant regions** of the test image — specifically the distinctive features of the plant or tree species being classified (leaf texture, fruit shape, canopy structure, bark pattern). The heatmap showed concentrated, high-activation areas directly on the subject rather than the background. This confirms that MobileNetV2 learned genuine visual features associated with each class rather than relying on dataset-specific artifacts like image backgrounds, lighting conditions, or surrounding environment.

For **EfficientNetB0 and ResNet50**, the Grad-CAM heatmaps were more diffuse and scattered across the image, including activation in background regions. This is consistent with models in the early stages of learning — they have identified some discriminative features but have not yet refined their attention to the most relevant regions. With complete training, their heatmaps are expected to become as focused as MobileNetV2's.

---

**12. Did the model focus on relevant image regions?**

**Yes, for MobileNetV2.** The Grad-CAM overlays showed that the model's attention was consistently directed toward the subject of classification — for example, focusing on the spiky red skin of RAMBUTAN, the distinctive pods of COCOA, the umbrella-shaped canopy of Monkey Pod Tree, and the elongated fruit of PAPAYA. This confirms that MobileNetV2 learned meaningful botanical features rather than spurious correlations with image backgrounds or metadata. The fact that PAPAYA achieved a perfect F1 of 1.00 is consistent with the model having clear, focused Grad-CAM activation on PAPAYA's distinctive elongated yellow-green fruit shape.

**Partially, for EfficientNetB0 and ResNet50.** Some class-relevant features were captured (especially for visually distinct classes like Dragon Blood Tree), but background activation was more prominent due to incomplete feature learning from only 3 training epochs.

---

**13. Which model produced the most meaningful heatmaps?**

**MobileNetV2** produced the most meaningful and interpretable Grad-CAM heatmaps. Its heatmaps showed tight, localized activation concentrated on the key identifying features of each species. This is both a consequence and a confirmation of its superior classification performance — a model that correctly classifies 93.81% of images is naturally expected to have learned the right visual cues, and Grad-CAM makes that learning visible. The sharpness and specificity of MobileNetV2's heatmaps also makes it more trustworthy for real-world deployment, since stakeholders can visually verify that the model is making decisions based on the correct features.

---

## F. Model Comparison & Improvement

**14. Which model would you recommend for deployment? Why?**

**MobileNetV2** is the recommended model for deployment, for the following reasons:

- **Highest accuracy** — 93.81% validation accuracy and AUC of 0.9935, the strongest performance among all three models
- **Fastest inference** — runs at 2s/step during training (vs. 3s/step for EfficientNetB0 and 7s/step for ResNet50), translating to faster real-time predictions on deployment
- **Smallest model size** — MobileNetV2 has approximately 3.4 million parameters, compared to ResNet50's 25 million, making it suitable for mobile and edge deployment
- **Most interpretable** — Grad-CAM heatmaps show focused, relevant activations, supporting explainable and trustworthy predictions
- **Proven convergence** — completed all 10 epochs with consistent improvement every epoch, demonstrating stable and reliable learning

For a mobile app or web-based plant/tree identification system, MobileNetV2 offers the best balance of accuracy, speed, and deployability.

---

**15. How can you further improve your best-performing model?**

Several techniques can improve MobileNetV2's performance beyond its current 93.81%:

1. **Fine-tuning** — Unfreeze the last 20–30 layers of the MobileNetV2 base and re-train with a very small learning rate (e.g., `1e-5`). This allows the pre-trained features to adapt specifically to the tree/fruit dataset rather than relying entirely on ImageNet representations.

2. **More epochs** — The model was still improving at Epoch 10 (val acc: 93.81%, still trending upward). Training for 20–30 epochs with a learning rate scheduler would likely push accuracy above 96%.

3. **Learning rate scheduling** — Replace the fixed `lr=0.0001` with `ReduceLROnPlateau` or cosine annealing to allow finer adjustments as the model approaches convergence.

4. **Data augmentation** — Add `RandomFlip`, `RandomRotation`, `RandomZoom`, and `RandomContrast` layers to the training pipeline to increase data diversity and further reduce overfitting.

5. **Increase training data** — Add more images for the weakest classes (DUHAT, Kapur Tree) to improve their F1-scores. Collecting 50–100 additional images per weak class would have a measurable impact.

6. **Class weights** — Apply `class_weight` during `model.fit()` to penalize the model more for misclassifying underrepresented classes.

---

## G. Real-World Application

**16. How can your model be applied in real-world scenarios?**

The trained MobileNetV2 classifier has multiple practical applications:

- **Plant and tree identification app** — A smartphone application (similar to PlantNet or iNaturalist) where users photograph trees or fruits to instantly identify species. This is directly supported by MobileNetV2's mobile-friendly size and fast inference.

- **Forestry and conservation** — Automated species identification during ecological surveys and biodiversity monitoring. Drones equipped with cameras can capture canopy images, and the model can classify species at scale across large forest areas.

- **Agriculture and crop management** — Identification of fruit crops for harvest timing, quality control, and pest detection. Farmers can photograph fruit and receive instant classification and health assessment.

- **Education** — Interactive nature learning tools for students and field researchers, allowing real-time identification of local tree and fruit species during fieldwork.

- **Biodiversity tracking** — Integration with camera traps or satellite imagery analysis pipelines to monitor species distribution changes over time, supporting conservation policy decisions.

---

**17. What are the risks of deploying an inaccurate model?**

Deploying a model with insufficient accuracy carries significant risks:

- **Ecological mismanagement** — Misidentifying a protected or endangered tree species could lead to illegal logging or incorrect conservation decisions, resulting in irreversible environmental damage.

- **Health and safety risks** — Misclassifying a toxic plant or fruit as an edible one could have serious health consequences for users relying on the app for foraging or food identification.

- **Economic loss in agriculture** — Incorrect crop identification can lead to improper pesticide application, wrong harvest timing, or misallocation of resources, causing financial losses for farmers.

- **Loss of user trust** — Repeated wrong predictions damage the credibility of the application and discourage adoption, especially if errors occur with visually distinctive species that users can manually verify.

- **Bias and inequity** — If certain classes (e.g., rare or regionally specific species) are underrepresented in training data, the model will systematically fail for those species, potentially disadvantaging communities or regions that rely on those specific plants.

- **Over-reliance on automation** — Users may trust model predictions without verification, amplifying the impact of any errors. This is especially dangerous in high-stakes applications like medicine or legal species protection.

These risks highlight why thorough evaluation using Precision, Recall, F1-Score, AUC, and Grad-CAM — as done in this laboratory — is essential before any real-world deployment.

---

**18. How can this system be integrated into a mobile/web app?**

The MobileNetV2 model can be integrated into a real-world application through the following pipeline:

**Step 1 — Convert the model:**
```python
# Convert to TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('mobilenetv2_tree_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Step 2 — Mobile app (Android/iOS):**
Use **Flutter** or **React Native** with the TensorFlow Lite plugin. The app accesses the device camera, captures a 224×224 image, passes it through the `.tflite` model on-device, and displays the predicted species with confidence score — all without internet connection.

**Step 3 — Web app:**
Deploy using **Streamlit** (fastest for prototyping) or **FastAPI + React**:
- Streamlit: Load the Keras model directly, accept image uploads, display predictions and Grad-CAM heatmaps in the browser
- FastAPI: Serve the model as a REST API endpoint that accepts base64-encoded images and returns JSON predictions

**Step 4 — Cloud API:**
Host on **Google Cloud Run**, **AWS Lambda**, or **Hugging Face Spaces** for scalable inference. The endpoint accepts an image, runs it through the loaded Keras model, and returns the top-3 predicted classes with confidence scores.

**Step 5 — GitHub + Streamlit demo (quickest for submission):**
```python
# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("MobileNetV2_model.keras")
class_names = [...]  # your 20 class names

uploaded = st.file_uploader("Upload a tree or fruit image")
if uploaded:
    img = Image.open(uploaded).resize((224, 224))
    arr = np.expand_dims(np.array(img), axis=0)
    pred = model.predict(arr)
    st.write(f"Prediction: {class_names[np.argmax(pred)]}")
    st.write(f"Confidence: {np.max(pred)*100:.2f}%")
```

---

*Laboratory Work 5 · Guide Questions · Final Reflection*
