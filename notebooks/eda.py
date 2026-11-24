"""
Exploratory Data Analysis (EDA) for the GCP Dataset.
Uses Pillow and Matplotlib (no OpenCV).
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TRAIN_DATASET_DIR, LABELS_FILE


def run_eda():
    output_dir = PROJECT_ROOT / "outputs" / "eda"
    os.makedirs(output_dir, exist_ok=True)

    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (AERIAL GCP)")
    print("=" * 60)

    print(f"\n1. BASIC STATISTICS")
    print(f"   Total annotations in json: {len(labels)}")

    shapes = Counter()
    missing_shape_count = 0
    xs, ys = [], []

    for k, v in labels.items():
        shape = v.get("verified_shape")
        if shape is not None:
            shapes[shape] += 1
        else:
            missing_shape_count += 1
            shapes["Unlabelled (default Cross)"] += 1

        if "mark" in v and "x" in v["mark"] and "y" in v["mark"]:
            xs.append(v["mark"]["x"])
            ys.append(v["mark"]["y"])

    print(f"   Shape distribution:")
    for shape, count in shapes.most_common():
        print(f"     {shape}: {count} ({100 * count / len(labels):.1f}%)")
    print(f"   Missing shape annotations: {missing_shape_count}")

    print(f"\n2. COORDINATE ANALYSIS")
    print(f"   X range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"   Y range: [{min(ys):.1f}, {max(ys):.1f}]")
    print(f"   X mean ± std: {np.mean(xs):.1f} ± {np.std(xs):.1f}")
    print(f"   Y mean ± std: {np.mean(ys):.1f} ± {np.std(ys):.1f}")

    print(f"\n3. PER-PROJECT ANALYSIS")
    project_stats = defaultdict(lambda: {"count": 0, "shapes": Counter()})
    for path, v in labels.items():
        project = path.split("/")[0]
        project_stats[project]["count"] += 1
        s = v.get("verified_shape", "Unlabelled")
        project_stats[project]["shapes"][s] += 1

    for project, stats in sorted(project_stats.items()):
        print(f"   {project}: {stats['count']} images | {dict(stats['shapes'])}")

    print(f"\n4. FILE EXISTENCE & DIMENSION CHECK")
    missing_files = []
    existing_files = []
    dim_counts = Counter()

    for rel_path in labels:
        full_path = Path(TRAIN_DATASET_DIR) / rel_path
        if full_path.exists():
            existing_files.append(rel_path)
            # Sample dimensions
            if len(existing_files) <= 100:
                try:
                    with Image.open(full_path) as img:
                        dim_counts[img.size] += 1
                except Exception:
                    pass
        else:
            missing_files.append(rel_path)

    print(f"   Files on disk matching JSON: {len(existing_files)}")
    print(f"   Files in JSON not on disk: {len(missing_files)}")
    print(f"   Sampled resolutions (width x height):")
    for dim, count in dim_counts.most_common(5):
        print(f"     {dim[0]}x{dim[1]}: {count} images")

    print(f"\n5. GENERATING VISUALIZATIONS...")

    # Plot 1: Shape Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    s_names = list(shapes.keys())
    s_vals = [shapes[s] for s in s_names]
    ax.bar(s_names, s_vals, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][:len(s_names)])
    ax.set_ylabel("Count")
    ax.set_title("Ground Truth Shape Distribution")
    for i, v in enumerate(s_vals):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shape_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Keypoint Locations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(xs, ys, alpha=0.3, s=12, color="#2980b9")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("GCP Keypoint Center Distribution")
    ax.invert_yaxis()

    ax = axes[1]
    h, xedges, yedges = np.histogram2d(xs, ys, bins=25)
    ax.imshow(h.T, origin="upper", cmap="magma", aspect="auto")
    ax.set_title("Spatial Heatmap of GCP Centers")
    ax.set_xlabel("X Bins")
    ax.set_ylabel("Y Bins")
    plt.tight_layout()
    plt.savefig(output_dir / "keypoint_spatial_distribution.png", dpi=150)
    plt.close()

    # Plot 3: Project breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    proj_names = sorted(project_stats.keys())
    proj_counts = [project_stats[p]["count"] for p in proj_names]
    ax.barh(proj_names, proj_counts, color="#1abc9c")
    ax.set_xlabel("Annotation Count")
    ax.set_title("Samples per Survey Project")
    for i, v in enumerate(proj_counts):
        ax.text(v + 1, i, str(v), va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "project_breakdown.png", dpi=150)
    plt.close()

    print(f"Visualizations successfully saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
