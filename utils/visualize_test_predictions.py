"""
Visual Verification Tool for Test Predictions.

Since test ground truth is hidden, this tool crops around the predicted (x, y)
coordinates on actual test images and overlays a red crosshair.
If the model correctly located the GCP, you will clearly see the physical marker
(Cross, Square, or L-Shape) directly under the crosshair.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DIR = Path(r"C:\Users\Pola Sai Prajwala\Downloads\test_dataset_extracted\test_dataset")
PRED_PATH = Path(r"C:\Users\Pola Sai Prajwala\.gemini\antigravity\scratch\skylark-gcp-estimation\outputs\predictions.json")
OUTPUT_DIR = Path(r"C:\Users\Pola Sai Prajwala\.gemini\antigravity\scratch\skylark-gcp-estimation\outputs\test_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PRED_PATH, "r") as f:
    preds = json.load(f)

# Sample 6 diverse test images from different projects
samples = list(preds.items())[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (rel_path, pred) in enumerate(samples):
    img_path = TEST_DIR / rel_path
    ax = axes[idx]
    
    if not img_path.exists():
        ax.set_title(f"Not found:\n{rel_path}", fontsize=8)
        ax.axis("off")
        continue

    px = pred["mark"]["x"]
    py = pred["mark"]["y"]
    shape = pred["verified_shape"]

    with Image.open(img_path) as img:
        w, h = img.size
        # Crop 400x400 window centered on predicted (px, py)
        crop_radius = 200
        left = max(0, int(px - crop_radius))
        top = max(0, int(py - crop_radius))
        right = min(w, int(px + crop_radius))
        bottom = min(h, int(py + crop_radius))

        crop = img.crop((left, top, right, bottom)).copy()
        draw = ImageDraw.Draw(crop)

        # Draw red crosshair at the predicted coordinate relative to crop
        cx = px - left
        cy = py - top
        r = 15
        draw.line([(cx - r, cy), (cx + r, cy)], fill="red", width=3)
        draw.line([(cx, cy - r), (cx, cy + r)], fill="red", width=3)
        draw.ellipse([(cx - 8, cy - 8), (cx + 8, cy + 8)], outline="red", width=2)

    ax.imshow(crop)
    short_name = Path(rel_path).name
    project = rel_path.split("/")[0]
    ax.set_title(f"{project}\n{short_name}\nPred: {shape} ({px:.1f}, {py:.1f})", fontsize=9)
    ax.axis("off")

plt.tight_layout()
save_file = OUTPUT_DIR / "sample_test_predictions_overlay.png"
plt.savefig(save_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visual verification to: {save_file}")
