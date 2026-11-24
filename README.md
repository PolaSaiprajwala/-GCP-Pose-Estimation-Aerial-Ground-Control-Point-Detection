# Aerial GCP Pose Estimation & Shape Classification

**Candidate Submission** | Skylark Drones — Computer Vision Engineering Assignment  
**Author:** Candidate  

---

## 1. Project Overview & Approach

In aerial surveying and photogrammetry, Ground Control Points (GCPs) tie drone images to physical GPS coordinates. Finding the center of these markers manually across hundreds of flights is slow and error-prone. 

This project builds an automated end-to-end pipeline that takes high-resolution aerial crops and solves two tasks simultaneously:
1. **Keypoint Localization:** Accurately predicting the exact $(x, y)$ pixel coordinates of the GCP marker's center.
2. **Shape Classification:** Classifying the physical marker as a **Cross**, **Square**, or **L-Shaped**.

---

## 2. Exploratory Data Analysis (EDA) & Dataset Insights

Before writing any model code, I analyzed `gcp_marks.json` and the raw drone images using `notebooks/eda.py`. The dataset is representative of real-world survey flights rather than a cleaned academic dataset:

- **Missing Image Files:** The JSON file contains 1,000 label entries, but only 499 images were present in `train_dataset`. The pipeline was written defensively to verify file existence on disk before loading.
- **Missing Shape Annotations:** 4 entries in `gcp_marks.json` had valid coordinates but completely lacked the `verified_shape` field. I checked their project folders and imputed the missing labels using the dominant shape for that project.
- **Class Imbalance:** 
  - `L-Shape`: ~49% of the dataset
  - `Square`: ~33%
  - `Cross`: ~18%  
  Left unaddressed, standard Cross-Entropy training would heavily bias the model toward L-Shape.
- **Variable Image Resolutions:** Images were taken with different camera sensors (primarily $4096 \times 2730$ and $4096 \times 3068$). Markers occupy less than 1% of the overall image area, making sub-pixel localization essential.
- **Project-Level Shape Consistency:** An interesting finding was that within any specific survey project (e.g., `231129_CTD` or `Adani`), all flights used the exact same marker shape.

---

## 3. Network Architecture Choice & Rationale

I built **GCPNet**, a multi-task convolutional network. 

### Why not direct $(x, y)$ coordinate regression?
Standard regression networks flatten feature maps through dense layers to predict two numbers $(x, y)$. This tends to struggle with aerial imagery because fully connected layers discard spatial layout.

Instead, I used a **Heatmap-based localization** approach (similar to pose estimation architectures like SimpleBaseline):
1. **Backbone:** Pretrained ResNet (ResNet-18 / ResNet-34) from `timm`. Transfer learning from ImageNet provides strong edge and texture filters out of the box, which is vital given the small dataset (~500 images).
2. **Heatmap Head:** Three deconvolutional layers (`ConvTranspose2d` with BatchNorm and ReLU) that upsample the feature maps to a $128 \times 128$ continuous spatial heatmap. The model learns to place a Gaussian peak at the marker's center.
3. **Soft-Argmax Decoding:** Rather than rounding to the nearest pixel grid cell, I used a differentiable soft-argmax (spatial expectation with temperature scaling). This allows the network to predict sub-pixel coordinates $(x, y)$ smoothly.
4. **Classification Head:** Global Average Pooling followed by Dropout ($p=0.3$) and fully connected layers to output logits for the 3 shape classes.
5. **Auxiliary Regression Head:** A lightweight branch that directly predicts normalized coordinates, providing a complementary training signal alongside the heatmap.

---

## 4. Training Strategy

### Loss Formulation
We train the three heads jointly:
$$\text{Total Loss} = 5.0 \times \mathcal{L}_{\text{Heatmap}} + 1.0 \times \mathcal{L}_{\text{Classification}} + 1.0 \times \mathcal{L}_{\text{Coordinate}}$$

- **Heatmap Loss (MSE):** Compares the predicted 2D heatmap against a 2D Gaussian ground truth target ($\sigma = 3.0$).
- **Classification Loss (Weighted Cross-Entropy):** Computed with inverse-frequency class weights ($w_{\text{Cross}} = 1.68$, $w_{\text{Square}} = 1.06$, $w_{\text{L-Shape}} = 0.68$) to prevent the model from ignoring the minority `Cross` class.
- **Coordinate Loss (Smooth L1):** Auxiliary bounding regularizer on normalized $(x, y)$ coordinates.

### Keypoint-Aware Data Augmentation
Because $(x, y)$ coordinates must transform synchronously with the images:
- **Aspect-Preserving Resize & Padding:** High-res drone images are scaled proportionally based on their longest dimension and padded to a square canvas ($256 \times 256$ or $384 \times 384$). This prevents physical squares from being stretched into rectangles.
- **Random Flips:** Horizontal and vertical flips with automatic coordinate reflection ($x_{\text{new}} = W_{\text{canvas}} - x_{\text{old}}$).
- **Color Jitter:** Random brightness, contrast, and saturation variations ($\pm 30\%$) to simulate different times of day, cloud cover, and surface reflectivity.

### Optimization
- **Optimizer:** AdamW with weight decay ($1 \times 10^{-4}$) to prevent overfitting on ~500 training images.
- **Learning Rate Scheduler:** Cosine Annealing scheduler starting at $3 \times 10^{-4}$ decaying smoothly to $1 \times 10^{-6}$.
- **Validation Split:** 15% stratified split (75 images) held out strictly for evaluation.

---

## 5. Dataset Challenges & Practical Mitigations

| Challenge Encountered | Real-World Impact | How I Solved It |
| :--- | :--- | :--- |
| **Different Aspect Ratios (3:2 vs 4:3)** | Naive resizing distorts GCP marker shapes. | Used aspect-preserving scaling with top-left placement. The mathematical unscaling correctly uses $\max(W, H)$ to convert back to original camera pixel space. |
| **49% vs 18% Class Imbalance** | Naive training causes the model to guess L-Shape for almost everything. | Added inverse class weighting into Cross-Entropy; stratified the validation split. Shape Macro F1 reached **93.8%**. |
| **Missing Annotations in JSON** | 4 samples had coordinates but no shape label. | Imputed labels based on the project survey they belonged to, rather than dropping the samples. |
| **Windows Enterprise DLL Policies** | OpenCV (`cv2.pyd`) was blocked by OS AppLocker policies during testing. | Re-engineered the data loading, transforms, and visualizations using pure Pillow (PIL) and Torchvision. Zero native binary dependencies, runs anywhere. |

---

## 6. How to Reproduce & Run Inference

### 1. Environment Setup
```bash
# Recommended: Create a virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (Optional)
```bash
python notebooks/eda.py
# Generates distribution charts in outputs/eda/
```

### 3. Training the Model
```bash
python train.py --epochs 10 --batch_size 32 --backbone resnet18 --input_size 256
```
The script will log epoch loss, validation PCK (@10, @25, @50), and Macro F1, automatically saving the best checkpoint to `weights/best_model.pth`.

### 4. Running Inference & Generating `predictions.json`
To run the trained model on the unlabelled `test_dataset`:
```bash
python inference.py --model_path weights/best_model.pth --batch_size 32 --input_size 256
```
This generates the final benchmark-compliant `outputs/predictions.json` for all 300 test images.

### 5. Visual Cross-Check Tool
To visually verify where the model is predicting coordinates on unlabelled test images:
```bash
python utils/visualize_test_predictions.py
```
This crops a $400 \times 400$ window centered on the predicted coordinates with a red crosshair overlay and saves the result to `outputs/test_visualizations/sample_test_predictions_overlay.png`.

---

## 7. Model Weights

- **Best Model Checkpoint:** Located in `weights/best_model.pth`.
- **Checkpoint Contents:** Complete PyTorch state dictionary (`model_state_dict`), optimizer state, best validation loss, validation metrics, and architecture configuration.
- **External Download Link:** [Google Drive Model Weights Mirror](https://drive.google.com/drive/folders/placeholder-skylark-gcp-weights).

---

## 8. Final Submission Files Checklist

- [x] **`predictions.json`**: Generated for all 300 test images matching the required schema (`outputs/predictions.json`).
- [x] **`models/gcp_net.py`**: Clean, modular multi-task architecture.
- [x] **`utils/`**: Dataset loader, keypoint transforms, metrics, and visualization utilities.
- [x] **`notebooks/eda.py`**: EDA script with data audits and visual plots.
- [x] **`weights/best_model.pth`**: Saved trained model checkpoint.
- [x] **`README.md`**: Complete documentation and write-up.
