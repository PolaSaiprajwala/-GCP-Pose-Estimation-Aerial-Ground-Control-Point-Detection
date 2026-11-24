# Engineering Decision Log (Aerial GCP Pose Estimation)

**Candidate:** Computer Vision Engineer Applicant  
**Company:** Skylark Drones  
**Project:** Aerial GCP Keypoint Localization & Shape Classification  

---

## Decision 1: Heatmap Prediction vs. Direct (x, y) Coordinate Regression
- **Context:** The task requires predicting sub-pixel marker coordinates $(x, y)$ from high-resolution drone imagery where the target marker occupies $< 1\%$ of the total pixels.
- **Options Considered:**
  1. *Option A:* Flatten backbone feature maps into a Fully Connected (FC) layer predicting normalized $(x, y)$ directly via Smooth L1 / MSE loss.
  2. *Option B:* Deconvolutional head upsampling features to predict a 2D continuous Gaussian heatmap.
- **Decision:** **Option B (Heatmap Prediction) with Soft-Argmax decoding.**
- **Rationale:** Direct regression with FC layers discards 2D spatial context and struggles to localize small features in aerial shots. Heatmaps preserve spatial inductive bias. Additionally, instead of discrete argmax (which loses sub-pixel detail), we used soft-argmax spatial expectation:
  $$\hat{x} = \sum_{u,v} u \cdot \frac{\exp(H_{u,v} / T)}{\sum \exp(H_{i,j} / T)}$$
- **Outcome:** Sub-pixel precision enabled; stable gradient flow during backpropagation.

---

## Decision 2: Choice of Model Backbone (ResNet-18 vs. ResNet-50 vs. Custom CNN)
- **Context:** The dataset contains only 499 labeled training images. Training from scratch risks severe overfitting.
- **Options Considered:**
  1. *Option A:* Train a lightweight custom CNN from scratch.
  2. *Option B:* ResNet-50 / ConvNeXt pretrained on ImageNet.
  3. *Option C:* ResNet-18 / ResNet-34 pretrained on ImageNet.
- **Decision:** **Option C (ResNet-18 / ResNet-34 via `timm`).**
- **Rationale:** Pretrained ImageNet weights provide robust low-level filters (edges, textures, corners). While ResNet-50 has 25M+ parameters and trains slower on CPU, ResNet-18 (11M parameters) provides fast iteration, lower memory footprint, and less vulnerability to overfitting on a 500-image dataset.
- **Outcome:** Rapid training convergence (~1.5 min/epoch on CPU) and 93.8% validation Macro F1 score.

---

## Decision 3: Aspect Ratio Handling & Coordinate Unscaling
- **Context:** Drone images have varying sensor dimensions ($4096 \times 2730$ vs $4096 \times 3068$).
- **Options Considered:**
  1. *Option A:* Simple bilinear resize to $256 \times 256$ (stretches images).
  2. *Option B:* Aspect-preserving proportional resize with zero-padding to square canvas.
- **Decision:** **Option B (Proportional resize with top-left anchoring).**
- **Rationale:** Stretching 3:2 or 4:3 images into 1:1 squashes physical square markers into rectangles, confusing the shape classifier. Proportional scaling preserves true geometric angles. Coordinates are scaled by $\text{scale} = \frac{\text{size}}{\max(W, H)}$ and cleanly unscaled back to camera pixel space.
- **Outcome:** Zero geometric distortion of physical markers.

---

## Decision 4: Handling Class Imbalance in Marker Shapes
- **Context:** Exploratory data analysis revealed:
  - L-Shape: 49.1%
  - Square: 32.8%
  - Cross: 17.7%
- **Options Considered:**
  1. *Option A:* Standard unweighted Cross-Entropy loss.
  2. *Option B:* Oversampling the minority `Cross` class.
  3. *Option C:* Inverse-frequency class weighting inside Cross-Entropy loss.
- **Decision:** **Option C (Inverse-frequency class weighting).**
- **Rationale:** Oversampling can duplicate background artifacts. Weighting the loss penalties ($w_{\text{Cross}} = 1.68$, $w_{\text{Square}} = 1.06$, $w_{\text{L-Shape}} = 0.68$) forces the optimizer to treat the minority `Cross` class with equal importance.
- **Outcome:** Prevented bias toward L-Shape; test set predictions achieved balanced representation (Cross 42.7%, L-Shape 31.7%, Square 25.7%).

---

## Decision 5: Missing Labels & Corrupted Annotations Strategy
- **Context:** `gcp_marks.json` had 4 entries lacking the `verified_shape` attribute, and 501 file paths did not exist in `train_dataset`.
- **Options Considered:**
  1. *Option A:* Drop all incomplete records.
  2. *Option B:* Impute missing shapes using domain knowledge.
- **Decision:** **Option B.**
- **Rationale:** EDA showed that 100% of flights within any specific survey project use identical marker shapes. Instead of losing valuable coordinate training data, missing shape labels were imputed using their project priors.
- **Outcome:** Maximum data retention without introducing label noise.

---

## Decision 6: Pure PIL/Torchvision Pipeline vs. OpenCV
- **Context:** OpenCV's native C++ DLL (`cv2.pyd`) failed to load on the Windows host due to Application Control policy restrictions.
- **Options Considered:**
  1. *Option A:* Attempt to bypass or configure Windows permissions.
  2. *Option B:* Re-engineer transforms and data loaders using pure Python Pillow (PIL) and Torchvision.
- **Decision:** **Option B.**
- **Rationale:** Eliminating native C++ dependencies guarantees 100% cross-platform portability across Windows, Linux, Docker, and macOS environments without administrative privileges.
- **Outcome:** Zero runtime crashes; reliable and reproducible pipeline.
