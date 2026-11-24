"""
Visualization utilities for training monitoring and result analysis using PIL and Matplotlib.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


def visualize_predictions(
    images, pred_coords, gt_coords, pred_shapes, gt_shapes,
    shape_names, save_path, num_samples=8
):
    """Visualize predicted vs ground truth keypoints on images."""
    n = min(num_samples, len(images))
    fig, axes = plt.subplots(2, max(1, n // 2), figsize=(4 * max(1, n // 2), 8))
    if n == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        if isinstance(images[i], str):
            with Image.open(images[i]) as img:
                w, h = img.size
                scale = 400 / max(w, h)
                img_disp = img.resize((int(w * scale), int(h * scale)))
                gt_x, gt_y = gt_coords[i] * scale
                pred_x, pred_y = pred_coords[i] * scale
        else:
            img_disp = images[i]
            gt_x, gt_y = gt_coords[i]
            pred_x, pred_y = pred_coords[i]

        ax.imshow(img_disp)
        ax.plot(gt_x, gt_y, "g+", markersize=14, markeredgewidth=2, label="GT")
        ax.plot(pred_x, pred_y, "rx", markersize=14, markeredgewidth=2, label="Pred")

        gt_shape = shape_names.get(gt_shapes[i], "?")
        pred_shape = shape_names.get(pred_shapes[i], "?")
        color = "green" if gt_shapes[i] == pred_shapes[i] else "red"
        ax.set_title(f"GT: {gt_shape} | Pred: {pred_shape}", fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path):
    """Plot training and validation curves."""
    num_plots = len(train_losses) + len(val_metrics)
    fig, axes = plt.subplots(1, max(1, num_plots), figsize=(5 * max(1, num_plots), 4))
    if num_plots == 1:
        axes = [axes]

    idx = 0
    for name in train_losses:
        ax = axes[idx]
        ax.plot(train_losses[name], label=f"Train {name}")
        if name in val_losses:
            ax.plot(val_losses[name], label=f"Val {name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1

    for name in val_metrics:
        ax = axes[idx]
        if name in train_metrics:
            ax.plot(train_metrics[name], label=f"Train {name}")
        ax.plot(val_metrics[name], label=f"Val {name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")
