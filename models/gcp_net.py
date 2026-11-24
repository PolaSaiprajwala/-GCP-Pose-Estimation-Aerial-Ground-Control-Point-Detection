"""
GCPNet: A multi-task network for GCP keypoint localization and shape classification.

Architecture:
- Backbone: ResNet-50 (pretrained on ImageNet) via timm
- Heatmap head: Deconvolution layers to predict a spatial heatmap for keypoint localization
- Classification head: Global Average Pool → FC layers for shape classification
- Direct regression head: FC layers for (x, y) coordinate regression (auxiliary)

Design Rationale:
- Heatmap-based localization is more accurate than direct regression for sub-pixel precision
- Multi-task learning shares the backbone, acting as a regularizer
- ResNet-50 provides a good balance of accuracy vs. speed/memory
- Deconv layers are standard in pose estimation (SimpleBaseline / HRNet style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HeatmapHead(nn.Module):
    """Deconvolution-based heatmap prediction head (SimpleBaseline style)."""

    def __init__(self, in_channels, out_size=128, num_keypoints=1):
        super().__init__()
        self.out_size = out_size
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.deconv_layers(x)
        heatmap = self.final_conv(x)
        # Resize to target heatmap size
        heatmap = F.interpolate(heatmap, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        return heatmap


class ClassificationHead(nn.Module):
    """Classification head for shape prediction."""

    def __init__(self, in_features, num_classes=3, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class RegressionHead(nn.Module):
    """Direct coordinate regression head (auxiliary)."""

    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # (x, y) normalized coordinates
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x):
        return self.head(x)


class GCPNet(nn.Module):
    """
    Multi-task GCP detection network.

    Outputs:
    - heatmap: (B, 1, H, H) spatial heatmap for keypoint localization
    - coords:  (B, 2) normalized (x, y) coordinates from regression head
    - logits:  (B, num_classes) classification logits for shape prediction
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        num_classes=3,
        heatmap_size=128,
    ):
        super().__init__()

        # Create backbone using timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],  # Last feature map only
        )

        # Get the number of channels from the last feature map
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy)
            feat_channels = features[-1].shape[1]

        # Task-specific heads
        self.heatmap_head = HeatmapHead(feat_channels, out_size=heatmap_size)
        self.classification_head = ClassificationHead(feat_channels, num_classes)
        self.regression_head = RegressionHead(feat_channels)

    def forward(self, x):
        features = self.backbone(x)
        feat = features[-1]  # (B, C, H/32, W/32) for ResNet

        heatmap = self.heatmap_head(feat)
        coords = self.regression_head(feat)
        logits = self.classification_head(feat)

        return {
            "heatmap": heatmap,
            "coords": coords,
            "logits": logits,
        }

    def predict_keypoint_from_heatmap(self, heatmap, original_w, original_h):
        """
        Extract (x, y) pixel coordinates from a heatmap prediction.
        Uses soft-argmax for sub-pixel accuracy.

        Args:
            heatmap: (B, 1, H, H) predicted heatmap
            original_w: original image width (or list)
            original_h: original image height (or list)

        Returns:
            coords: (B, 2) pixel coordinates in original image space
        """
        B, _, H, W = heatmap.shape
        # Soft-argmax
        heatmap_flat = heatmap.view(B, -1)
        heatmap_softmax = F.softmax(heatmap_flat / 0.1, dim=-1)  # Temperature scaling
        heatmap_softmax = heatmap_softmax.view(B, 1, H, W)

        # Create coordinate grids
        device = heatmap.device
        grid_y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(B, 1, H, W)
        grid_x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W).expand(B, 1, H, W)

        # Weighted average of coordinates
        pred_x = (heatmap_softmax * grid_x).sum(dim=[2, 3]) / W  # Normalized [0, 1]
        pred_y = (heatmap_softmax * grid_y).sum(dim=[2, 3]) / H  # Normalized [0, 1]

        # Scale to original image dimensions based on LongestMaxSize transform
        if isinstance(original_w, (int, float)):
            max_dim = max(float(original_w), float(original_h))
            pred_x = pred_x * max_dim
            pred_y = pred_y * max_dim
        else:
            # Batch of different sizes
            original_w = torch.tensor(original_w, device=device, dtype=torch.float32).view(B, 1)
            original_h = torch.tensor(original_h, device=device, dtype=torch.float32).view(B, 1)
            max_dim = torch.maximum(original_w, original_h)
            pred_x = pred_x * max_dim
            pred_y = pred_y * max_dim

        return torch.cat([pred_x, pred_y], dim=-1)
