"""
PyTorch / PIL Data Augmentation pipeline for GCP keypoints and images.
Pure Python + Torchvision implementation with no OpenCV dependency.
"""

import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image


class GCPTransform:
    """
    Transforms an image and keypoint together while keeping aspect ratio.
    Pads to (input_size, input_size).
    """

    def __init__(self, input_size=512, is_train=True):
        self.input_size = input_size
        self.is_train = is_train
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img: Image.Image, mark_x: float, mark_y: float):
        orig_w, orig_h = img.size
        max_dim = max(orig_w, orig_h)
        scale = self.input_size / max_dim
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Resize preserving aspect ratio
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad to (input_size, input_size) with top-left placement
        canvas = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))
        canvas.paste(img_resized, (0, 0))

        # Transformed coordinates
        tx = mark_x * scale
        ty = mark_y * scale

        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                tx = self.input_size - tx

            # Random Vertical Flip
            if random.random() > 0.5:
                canvas = canvas.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                ty = self.input_size - ty

            # Color Jitter
            if random.random() > 0.3:
                canvas = self.color_jitter(canvas)

        # Convert to tensor
        tensor_img = TF.to_tensor(canvas)
        tensor_img = self.normalize(tensor_img)

        # Normalized coordinates [0, 1] relative to input_size
        norm_x = max(0.0, min(1.0, tx / self.input_size))
        norm_y = max(0.0, min(1.0, ty / self.input_size))

        return tensor_img, norm_x, norm_y


def get_train_transforms(input_size=512):
    return GCPTransform(input_size=input_size, is_train=True)


def get_val_transforms(input_size=512):
    return GCPTransform(input_size=input_size, is_train=False)
