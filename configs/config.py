"""
Configuration for the GCP Pose Estimation pipeline.
All hyperparameters and paths are centralized here.
"""
import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
# Update these to match your local directory structure
TRAIN_DATASET_DIR = r"C:\Users\Pola Sai Prajwala\Downloads\train_dataset-20261124T142201Z-1-002\train_dataset"
TEST_DATASET_DIR = r"C:\Users\Pola Sai Prajwala\Downloads\test_dataset_extracted\test_dataset"
LABELS_FILE = os.path.join(TRAIN_DATASET_DIR, "gcp_marks.json")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# ============================================================
# Image Configuration
# ============================================================
# Input size for the model
INPUT_SIZE = 384  # Fast, highly accurate resolution for aerial GCP detection
ORIGINAL_WIDTH = None   # Set dynamically per image
ORIGINAL_HEIGHT = None  # Set dynamically per image

# ============================================================
# Model Configuration
# ============================================================
BACKBONE = "resnet34"  # ResNet-34 offers great accuracy-efficiency balance
PRETRAINED = True
NUM_CLASSES = 3  # Cross, Square, L-Shape
NUM_KEYPOINTS = 1  # Single center point (x, y)

# Shape class mapping
SHAPE_CLASSES = {
    "Cross": 0,
    "Square": 1,
    "L-Shape": 2,
    "L-Shaped": 2,  # Handle both naming conventions
}
SHAPE_NAMES = {0: "Cross", 1: "Square", 2: "L-Shape"}

# ============================================================
# Training Configuration
# ============================================================
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER = "cosine"  # Options: cosine, step, plateau

# Loss weights: balance between keypoint regression and classification
KEYPOINT_LOSS_WEIGHT = 1.0
CLASSIFICATION_LOSS_WEIGHT = 1.0
HEATMAP_LOSS_WEIGHT = 5.0  # Heatmap-based keypoint loss weight

# Heatmap configuration
USE_HEATMAP = True
HEATMAP_SIZE = 128  # Output heatmap resolution
HEATMAP_SIGMA = 3.0  # Gaussian sigma for heatmap generation

# ============================================================
# Data Augmentation
# ============================================================
# Training augmentations
AUGMENTATION = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.3,
    "rotate_limit": 30,
    "rotate_p": 0.5,
    "brightness_contrast_p": 0.4,
    "blur_p": 0.2,
    "noise_p": 0.2,
    "color_jitter_p": 0.3,
    "random_crop_scale": (0.7, 1.0),  # For random resized crop
}

# ============================================================
# Validation / Training Split
# ============================================================
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# ============================================================
# Early Stopping
# ============================================================
EARLY_STOPPING_PATIENCE = 15

# ============================================================
# Device
# ============================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
PIN_MEMORY = DEVICE == "cuda"
