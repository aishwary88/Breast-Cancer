"""
model.py
--------
DenseNet121 model factory for breast cancer binary classification.

DenseNet121 architecture:
    - 121 layers with dense connections (each layer receives feature maps
      from ALL preceding layers)
    - Pretrained on ImageNet (1.2M images, 1000 classes)
    - Final classifier replaced with Dropout + Linear(1024 → 2)
    - ~7M trainable parameters — lighter than ResNet50, strong on medical imaging

Why DenseNet121 for mammography?
    - Dense connections encourage feature reuse — important for subtle lesion patterns
    - Widely used in medical imaging benchmarks (CheXNet, etc.)
    - Good accuracy/parameter tradeoff for 4GB VRAM
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    num_classes:     int  = 2,
    pretrained:      bool = True,
    freeze_backbone: bool = False,
    dropout:         float = 0.4,
) -> nn.Module:
    """
    Build DenseNet121 with a custom binary classification head.

    Parameters
    ----------
    num_classes     : number of output classes (2 for benign/malignant)
    pretrained      : load ImageNet pretrained weights
    freeze_backbone : if True, only the classifier head is trained
    dropout         : dropout rate before the final linear layer

    Returns
    -------
    nn.Module ready for training
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model   = models.densenet121(weights=weights)

    # ── replace the classifier head ───────────────────────────────────────
    # Original: Linear(1024 → 1000)
    # Ours:     Dropout → Linear(1024 → 2)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    # ── optionally freeze the feature extractor ───────────────────────────
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }
