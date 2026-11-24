"""
Evaluation metrics for the GCP detection pipeline.

- PCK (Percentage of Correct Keypoints) at various thresholds
- Macro F1 Score for shape classification
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def compute_pck(pred_coords, gt_coords, thresholds=[10, 25, 50, 100]):
    """
    Compute Percentage of Correct Keypoints (PCK).
    
    Args:
        pred_coords: (N, 2) predicted pixel coordinates
        gt_coords: (N, 2) ground truth pixel coordinates
        thresholds: list of pixel distance thresholds
    
    Returns:
        dict: {threshold: pck_percentage}
    """
    pred_coords = np.array(pred_coords)
    gt_coords = np.array(gt_coords)
    
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=1))
    
    results = {}
    for t in thresholds:
        pck = np.mean(distances <= t) * 100
        results[f"PCK@{t}px"] = pck
    
    results["mean_distance"] = np.mean(distances)
    results["median_distance"] = np.median(distances)
    
    return results


def compute_macro_f1(pred_labels, gt_labels, class_names=None):
    """
    Compute Macro F1 Score for shape classification.
    
    Args:
        pred_labels: (N,) predicted class indices
        gt_labels: (N,) ground truth class indices
        class_names: optional list of class names for the report
    
    Returns:
        dict with macro_f1 and per-class metrics
    """
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)
    
    macro_f1 = f1_score(gt_labels, pred_labels, average="macro")
    
    results = {
        "macro_f1": macro_f1,
    }
    
    if class_names:
        report = classification_report(
            gt_labels, pred_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        results["per_class"] = report
        
        cm = confusion_matrix(gt_labels, pred_labels)
        results["confusion_matrix"] = cm.tolist()
    
    return results
