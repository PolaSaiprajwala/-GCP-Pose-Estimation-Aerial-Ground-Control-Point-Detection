"""
Inference script for the GCP Pose Estimation pipeline.

Loads the best trained model, runs inference on the test dataset,
and produces predictions.json in the required format.

Usage:
    python inference.py
    python inference.py --model_path weights/best_model.pth --test_dir /path/to/test_dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import *
from models.gcp_net import GCPNet
from utils.dataset import GCPTestDataset
from utils.augmentations import get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test dataset")
    parser.add_argument("--model_path", type=str, default=str(WEIGHTS_DIR / "best_model.pth"))
    parser.add_argument("--test_dir", type=str, default=TEST_DATASET_DIR)
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_DIR / "predictions.json"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)
    return parser.parse_args()


def test_collate_fn(batch):
    """Custom collate for test dataset (no labels)."""
    images, metadata = zip(*batch)
    images = torch.stack(images)
    return images, list(metadata)


def main():
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print("=" * 60)
    print("GCP Pose Estimation - Inference")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Test dir: {args.test_dir}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        print("Please train the model first using: python train.py")
        sys.exit(1)
    
    checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    
    # Reconstruct model
    model = GCPNet(
        backbone_name=config.get("backbone", BACKBONE),
        pretrained=False,
        num_classes=config.get("num_classes", NUM_CLASSES),
        heatmap_size=config.get("heatmap_size", HEATMAP_SIZE),
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    if "val_metrics" in checkpoint:
        vm = checkpoint["val_metrics"]
        print(f"  Val loss: {vm.get('total_loss', '?'):.4f}")
        print(f"  PCK@25: {vm.get('PCK@25px', '?')}")
        print(f"  Macro F1: {vm.get('macro_f1', '?'):.3f}")
    print()
    
    # Data
    transform = get_val_transforms(args.input_size)
    test_dataset = GCPTestDataset(args.test_dir, transform=transform, input_size=args.input_size)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=test_collate_fn,
    )
    
    # Reverse shape mapping for output
    idx_to_shape = {0: "Cross", 1: "Square", 2: "L-Shape"}
    
    # Run inference
    predictions = {}
    
    with torch.no_grad():
        for images, metadata in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            
            outputs = model(images)
            
            # Get keypoint coordinates from heatmap
            orig_ws = [m["orig_w"] for m in metadata]
            orig_hs = [m["orig_h"] for m in metadata]
            
            pred_coords = model.predict_keypoint_from_heatmap(
                outputs["heatmap"], orig_ws, orig_hs
            ).cpu().numpy()
            
            # Get shape predictions
            pred_shapes = outputs["logits"].argmax(dim=1).cpu().numpy()
            
            # Build prediction entries
            for i, meta in enumerate(metadata):
                rel_path = meta["rel_path"]
                x = float(pred_coords[i, 0])
                y = float(pred_coords[i, 1])
                shape = idx_to_shape[int(pred_shapes[i])]
                
                predictions[rel_path] = {
                    "mark": {
                        "x": x,
                        "y": y,
                    },
                    "verified_shape": shape,
                }
    
    # Save predictions
    with open(args.output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"\nSaved {len(predictions)} predictions to {args.output_path}")
    
    # Print shape distribution
    from collections import Counter
    shape_counts = Counter(p["verified_shape"] for p in predictions.values())
    print(f"\nPredicted shape distribution:")
    for shape, count in shape_counts.most_common():
        print(f"  {shape}: {count} ({100 * count / len(predictions):.1f}%)")


if __name__ == "__main__":
    main()
