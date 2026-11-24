"""
PyTorch Dataset for GCP keypoint + classification using PIL and Torchvision.
"""

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import SHAPE_CLASSES, HEATMAP_SIZE, HEATMAP_SIGMA, INPUT_SIZE


def generate_heatmap(cx, cy, heatmap_size, sigma=3.0):
    """
    Generate a 2D Gaussian heatmap centered at (cx, cy).
    cx, cy in normalized [0, 1] coordinates.
    """
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    px = cx * heatmap_size
    py = cy * heatmap_size

    if px < 0 or px >= heatmap_size or py < 0 or py >= heatmap_size:
        return heatmap

    x = np.arange(heatmap_size, dtype=np.float32)
    y = np.arange(heatmap_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    heatmap = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * (sigma ** 2)))
    return heatmap


class GCPDataset(Dataset):
    """Dataset for training/validation of GCP markers."""

    def __init__(
        self,
        data_dir,
        labels_dict,
        transform=None,
        heatmap_size=HEATMAP_SIZE,
        heatmap_sigma=HEATMAP_SIGMA,
        input_size=INPUT_SIZE,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.input_size = input_size

        self.samples = []
        self.shape_labels = []

        for rel_path, annotation in labels_dict.items():
            img_path = self.data_dir / rel_path
            if img_path.exists() and "mark" in annotation:
                shape_name = annotation.get("verified_shape")
                # Default to Cross if shape is unlabelled
                if shape_name is None:
                    shape_name = "Cross"

                shape_label = SHAPE_CLASSES.get(shape_name, -1)
                if shape_label == -1:
                    continue

                self.samples.append({
                    "rel_path": rel_path,
                    "img_path": str(img_path),
                    "mark_x": float(annotation["mark"]["x"]),
                    "mark_y": float(annotation["mark"]["y"]),
                    "shape_label": shape_label,
                    "shape_name": shape_name,
                })
                self.shape_labels.append(shape_label)

        print(f"Loaded {len(self.samples)} valid samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with Image.open(sample["img_path"]) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            if self.transform:
                image_tensor, norm_x, norm_y = self.transform(
                    img, sample["mark_x"], sample["mark_y"]
                )
            else:
                max_dim = max(orig_w, orig_h)
                scale = self.input_size / max_dim
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))
                canvas.paste(resized, (0, 0))

                image_tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
                norm_x = (sample["mark_x"] * scale) / self.input_size
                norm_y = (sample["mark_y"] * scale) / self.input_size

        heatmap = generate_heatmap(norm_x, norm_y, self.heatmap_size, self.heatmap_sigma)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)

        coords = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        shape_label = torch.tensor(sample["shape_label"], dtype=torch.long)

        metadata = {
            "rel_path": sample["rel_path"],
            "orig_w": orig_w,
            "orig_h": orig_h,
            "mark_x": sample["mark_x"],
            "mark_y": sample["mark_y"],
        }

        return image_tensor, heatmap, coords, shape_label, metadata


class GCPTestDataset(Dataset):
    """Test dataset without ground truth."""

    def __init__(self, data_dir, transform=None, input_size=INPUT_SIZE):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.input_size = input_size

        self.image_paths = sorted(
            list(self.data_dir.rglob("*.JPG"))
            + list(self.data_dir.rglob("*.jpg"))
            + list(self.data_dir.rglob("*.jpeg"))
        )
        print(f"Found {len(self.image_paths)} test images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        rel_path = str(img_path.relative_to(self.data_dir)).replace("\\", "/")

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            if self.transform:
                image_tensor, _, _ = self.transform(img, 0.0, 0.0)
            else:
                max_dim = max(orig_w, orig_h)
                scale = self.input_size / max_dim
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))
                canvas.paste(resized, (0, 0))
                image_tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0

        metadata = {
            "rel_path": rel_path,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }

        return image_tensor, metadata


def get_train_val_datasets(data_dir, labels_file, train_transform, val_transform, val_split=0.15, seed=42):
    """Stratified train and validation dataset creation."""
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)

    full_dataset = GCPDataset(data_dir, labels_dict, transform=None)
    labels = full_dataset.shape_labels

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(range(len(labels)), labels))

    train_labels = {}
    val_labels = {}
    all_samples = full_dataset.samples

    for i in train_idx:
        s = all_samples[i]
        train_labels[s["rel_path"]] = {
            "mark": {"x": s["mark_x"], "y": s["mark_y"]},
            "verified_shape": s["shape_name"],
        }

    for i in val_idx:
        s = all_samples[i]
        val_labels[s["rel_path"]] = {
            "mark": {"x": s["mark_x"], "y": s["mark_y"]},
            "verified_shape": s["shape_name"],
        }

    train_dataset = GCPDataset(data_dir, train_labels, transform=train_transform)
    val_dataset = GCPDataset(data_dir, val_labels, transform=val_transform)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    return train_dataset, val_dataset


def get_test_dataset(data_dir, transform=None, input_size=INPUT_SIZE):
    """Create test dataset for inference."""
    return GCPTestDataset(data_dir, transform=transform, input_size=input_size)

