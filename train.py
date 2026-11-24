"""
Training script for the GCP Pose Estimation pipeline.

Multi-task training with:
1. Heatmap MSE loss for keypoint localization
2. Cross-entropy loss for shape classification
3. Smooth L1 loss for direct coordinate regression (auxiliary)

Usage:
    python train.py
    python train.py --epochs 150 --batch_size 8 --backbone resnet50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import *
from models.gcp_net import GCPNet
from utils.dataset import get_train_val_datasets
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.metrics import compute_pck, compute_macro_f1
from utils.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train GCP Pose Estimation Model")
    parser.add_argument("--train_dir", type=str, default=TRAIN_DATASET_DIR)
    parser.add_argument("--labels_file", type=str, default=LABELS_FILE)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)
    parser.add_argument("--heatmap_size", type=int, default=HEATMAP_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--weights_dir", type=str, default=str(WEIGHTS_DIR))
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights for imbalanced classes."""
    from collections import Counter
    counts = Counter(dataset.shape_labels)
    total = len(dataset.shape_labels)
    num_classes = len(counts)
    
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * c))
    
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {weights.tolist()}")
    return weights


def train_one_epoch(model, loader, optimizer, criterion_heatmap, criterion_cls, criterion_reg, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    heatmap_loss_sum = 0
    cls_loss_sum = 0
    reg_loss_sum = 0
    
    all_pred_shapes = []
    all_gt_shapes = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, heatmaps, coords, shapes, metadata in pbar:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        coords = coords.to(device)
        shapes = shapes.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Losses
        loss_heatmap = criterion_heatmap(outputs["heatmap"], heatmaps)
        loss_cls = criterion_cls(outputs["logits"], shapes)
        loss_reg = criterion_reg(outputs["coords"], coords)
        
        loss = (
            HEATMAP_LOSS_WEIGHT * loss_heatmap
            + CLASSIFICATION_LOSS_WEIGHT * loss_cls
            + KEYPOINT_LOSS_WEIGHT * loss_reg
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        heatmap_loss_sum += loss_heatmap.item()
        cls_loss_sum += loss_cls.item()
        reg_loss_sum += loss_reg.item()
        
        pred_shapes = outputs["logits"].argmax(dim=1).cpu().numpy()
        all_pred_shapes.extend(pred_shapes)
        all_gt_shapes.extend(shapes.cpu().numpy())
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "hm": f"{loss_heatmap.item():.4f}",
            "cls": f"{loss_cls.item():.4f}",
        })
    
    n = len(loader)
    metrics = {
        "total_loss": total_loss / n,
        "heatmap_loss": heatmap_loss_sum / n,
        "cls_loss": cls_loss_sum / n,
        "reg_loss": reg_loss_sum / n,
        "cls_accuracy": np.mean(np.array(all_pred_shapes) == np.array(all_gt_shapes)),
    }
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion_heatmap, criterion_cls, criterion_reg, device, input_size):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    heatmap_loss_sum = 0
    cls_loss_sum = 0
    reg_loss_sum = 0
    
    all_pred_coords_orig = []
    all_gt_coords_orig = []
    all_pred_shapes = []
    all_gt_shapes = []
    
    for images, heatmaps, coords, shapes, metadata in tqdm(loader, desc="[Val]", leave=False):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        coords = coords.to(device)
        shapes = shapes.to(device)
        
        outputs = model(images)
        
        loss_heatmap = criterion_heatmap(outputs["heatmap"], heatmaps)
        loss_cls = criterion_cls(outputs["logits"], shapes)
        loss_reg = criterion_reg(outputs["coords"], coords)
        
        loss = (
            HEATMAP_LOSS_WEIGHT * loss_heatmap
            + CLASSIFICATION_LOSS_WEIGHT * loss_cls
            + KEYPOINT_LOSS_WEIGHT * loss_reg
        )
        
        total_loss += loss.item()
        heatmap_loss_sum += loss_heatmap.item()
        cls_loss_sum += loss_cls.item()
        reg_loss_sum += loss_reg.item()
        
        # Decode keypoint from heatmap
        orig_ws = [m["orig_w"] for m in metadata]
        orig_hs = [m["orig_h"] for m in metadata]
        
        pred_pixel_coords = model.predict_keypoint_from_heatmap(
            outputs["heatmap"], orig_ws, orig_hs
        ).cpu().numpy()
        
        gt_pixel_coords = np.stack([
            np.array([m["mark_x"] for m in metadata]),
            np.array([m["mark_y"] for m in metadata]),
        ], axis=-1)
        
        all_pred_coords_orig.append(pred_pixel_coords)
        all_gt_coords_orig.append(gt_pixel_coords)
        
        pred_shapes = outputs["logits"].argmax(dim=1).cpu().numpy()
        all_pred_shapes.extend(pred_shapes)
        all_gt_shapes.extend(shapes.cpu().numpy())
    
    n = len(loader)
    
    # Compute PCK on original pixel coordinates
    all_pred_coords_orig = np.concatenate(all_pred_coords_orig, axis=0)
    all_gt_coords_orig = np.concatenate(all_gt_coords_orig, axis=0)
    
    pck_results = compute_pck(all_pred_coords_orig, all_gt_coords_orig)
    f1_results = compute_macro_f1(all_pred_shapes, all_gt_shapes, ["Cross", "Square", "L-Shape"])
    
    metrics = {
        "total_loss": total_loss / n,
        "heatmap_loss": heatmap_loss_sum / n,
        "cls_loss": cls_loss_sum / n,
        "reg_loss": reg_loss_sum / n,
        "cls_accuracy": np.mean(np.array(all_pred_shapes) == np.array(all_gt_shapes)),
        **pck_results,
        "macro_f1": f1_results["macro_f1"],
    }
    
    return metrics


def custom_collate_fn(batch):
    """Custom collate to handle metadata dicts."""
    images, heatmaps, coords, shapes, metadata = zip(*batch)
    
    images = torch.stack(images)
    heatmaps = torch.stack(heatmaps)
    coords = torch.stack(coords)
    shapes = torch.stack(shapes)
    # metadata stays as a list of dicts
    
    return images, heatmaps, coords, shapes, list(metadata)


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    print("=" * 60)
    print("GCP Pose Estimation - Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Backbone: {args.backbone}")
    print(f"Input size: {args.input_size}")
    print(f"Heatmap size: {args.heatmap_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Data
    train_transform = get_train_transforms(args.input_size)
    val_transform = get_val_transforms(args.input_size)
    
    train_dataset, val_dataset = get_train_val_datasets(
        args.train_dir, args.labels_file,
        train_transform, val_transform,
        val_split=VAL_SPLIT, seed=RANDOM_SEED,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=custom_collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=custom_collate_fn,
    )
    
    # Model
    model = GCPNet(
        backbone_name=args.backbone,
        pretrained=PRETRAINED,
        num_classes=NUM_CLASSES,
        heatmap_size=args.heatmap_size,
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    criterion_heatmap = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_reg = nn.SmoothL1Loss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    if SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    best_pck_25 = 0
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    history = {
        "train_loss": [], "val_loss": [],
        "train_hm_loss": [], "val_hm_loss": [],
        "train_cls_loss": [], "val_cls_loss": [],
        "val_pck10": [], "val_pck25": [], "val_pck50": [],
        "val_f1": [],
    }
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            criterion_heatmap, criterion_cls, criterion_reg,
            DEVICE, epoch,
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_heatmap, criterion_cls, criterion_reg,
            DEVICE, args.input_size,
        )
        
        # Update scheduler
        if SCHEDULER == "plateau":
            scheduler.step(val_metrics["total_loss"])
        else:
            scheduler.step()
        
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        
        # Log
        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"PCK@10: {val_metrics.get('PCK@10px', 0):.1f}% | "
            f"PCK@25: {val_metrics.get('PCK@25px', 0):.1f}% | "
            f"PCK@50: {val_metrics.get('PCK@50px', 0):.1f}% | "
            f"F1: {val_metrics['macro_f1']:.3f} | "
            f"LR: {lr:.2e}"
        )
        
        # Save history
        history["train_loss"].append(train_metrics["total_loss"])
        history["val_loss"].append(val_metrics["total_loss"])
        history["train_hm_loss"].append(train_metrics["heatmap_loss"])
        history["val_hm_loss"].append(val_metrics["heatmap_loss"])
        history["train_cls_loss"].append(train_metrics["cls_loss"])
        history["val_cls_loss"].append(val_metrics["cls_loss"])
        history["val_pck10"].append(val_metrics.get("PCK@10px", 0))
        history["val_pck25"].append(val_metrics.get("PCK@25px", 0))
        history["val_pck50"].append(val_metrics.get("PCK@50px", 0))
        history["val_f1"].append(val_metrics["macro_f1"])
        
        # Save best model (by combined score: PCK@25 + F1)
        combined_score = val_metrics.get("PCK@25px", 0) / 100 + val_metrics["macro_f1"]
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "config": {
                    "backbone": args.backbone,
                    "input_size": args.input_size,
                    "heatmap_size": args.heatmap_size,
                    "num_classes": NUM_CLASSES,
                },
            }
            torch.save(checkpoint, os.path.join(args.weights_dir, "best_model.pth"))
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, os.path.join(args.weights_dir, "latest_checkpoint.pth"))
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_training_curves(
        train_losses={"Total Loss": history["train_loss"], "Heatmap Loss": history["train_hm_loss"]},
        val_losses={"Total Loss": history["val_loss"], "Heatmap Loss": history["val_hm_loss"]},
        train_metrics={},
        val_metrics={"PCK@25px": history["val_pck25"], "Macro F1": history["val_f1"]},
        save_path=os.path.join(args.output_dir, "training_curves.png"),
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.weights_dir, 'best_model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
