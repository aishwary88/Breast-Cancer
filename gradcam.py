"""
gradcam.py
----------
Grad-CAM for ConvNeXt-Tiny breast cancer classifier.

Matches EXACTLY the architecture in app.py:
  - Grayscale input (1 channel)
  - ConvNeXt-Tiny backbone
  - Binary output (sigmoid, 1 neuron)
  - Checkpoint: best_fold_0.pth

Target layer: model.features[7]  (last ConvNeXt stage)

Usage (standalone):
    py -3.12 gradcam.py --image sample_malignant.jpg
    py -3.12 gradcam.py --image sample_benign.jpg
    py -3.12 gradcam.py --image sample_malignant.jpg --save gradcam_result.png

Also importable by app.py:
    from gradcam import GradCAM, generate_gradcam_figure
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Config — must match app.py exactly
# ─────────────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 448
CHECKPOINT = "best_fold_0.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Build model — identical to app.py
# ─────────────────────────────────────────────────────────────────────────────

def build_convnext() -> nn.Module:
    """
    Recreate the exact same ConvNeXt-Tiny architecture used in training.
    Must match app.py perfectly for weights to load correctly.
    """
    model = models.convnext_tiny(weights=None)

    # Grayscale input: replace first conv (3ch → 1ch)
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=96,
        kernel_size=4,
        stride=4,
    )

    # Binary classifier head: replace final linear (768 → 1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )

    return model


def load_model(checkpoint_path: str = CHECKPOINT) -> nn.Module:
    """Load ConvNeXt-Tiny with trained weights."""
    model = build_convnext().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — identical to app.py
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img: Image.Image) -> Image.Image:
    """Apply CLAHE contrast enhancement (same as app.py)."""
    arr   = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    arr   = clahe.apply(arr)
    return Image.fromarray(arr)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess(image_path: str) -> tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess a mammogram image.

    Returns
    -------
    tensor    : (1, 1, 448, 448) ready for model
    orig_img  : original PIL image (grayscale, before CLAHE) for display
    """
    img      = Image.open(image_path).convert("L")
    orig_img = img.copy()
    img      = apply_clahe(img)
    tensor   = transform(img).unsqueeze(0).to(DEVICE)
    return tensor, orig_img


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for ConvNeXt-Tiny.

    Target layer: model.features[7]
    This is the last ConvNeXt stage — highest-level spatial features.
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        # Last ConvNeXt stage = features[7]
        target = self.model.features[7]

        def fwd_hook(module, inp, out):
            # ConvNeXt uses (B, H, W, C) — permute to (B, C, H, W)
            self.activations = out.permute(0, 3, 1, 2).detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].permute(0, 3, 1, 2).detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self, tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """
        Generate Grad-CAM heatmap for the malignant class.

        Parameters
        ----------
        tensor : (1, 1, H, W) preprocessed image tensor

        Returns
        -------
        cam         : (H, W) heatmap, values in [0, 1]
        probability : malignant probability (sigmoid output)
        """
        self.model.eval()
        tensor = tensor.requires_grad_(True)

        # Forward pass
        output      = self.model(tensor)                    # (1, 1)
        probability = torch.sigmoid(output).item()

        # Backprop w.r.t. the single output neuron
        self.model.zero_grad()
        output.backward()

        # Global average pool gradients over spatial dims → weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=(tensor.shape[2], tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, probability


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmap_overlay(
    orig_img: Image.Image,
    cam:      np.ndarray,
    alpha:    float = 0.45,
) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Returns uint8 RGB numpy array.
    """
    # Resize original to match cam size
    orig_rgb = np.array(orig_img.resize((cam.shape[1], cam.shape[0])))
    orig_rgb = np.stack([orig_rgb] * 3, axis=-1)   # grayscale → RGB

    # Colormap heatmap
    heatmap = (plt.cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (alpha * heatmap + (1 - alpha) * orig_rgb).astype(np.uint8)
    return overlay


def generate_gradcam_figure(
    orig_img:    Image.Image,
    cam:         np.ndarray,
    probability: float,
) -> plt.Figure:
    """
    Build a matplotlib figure with 3 panels:
        Original | Heatmap | Overlay

    Used by both standalone script and Streamlit app.
    """
    prediction = "MALIGNANT" if probability > 0.5 else "BENIGN"
    color      = "#e74c3c" if probability > 0.5 else "#2ecc71"

    orig_arr = np.array(orig_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    heatmap  = (plt.cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay  = make_heatmap_overlay(orig_img.resize((IMAGE_SIZE, IMAGE_SIZE)), cam)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")

    titles = ["Original Mammogram", "Grad-CAM Heatmap", "Overlay"]
    imgs   = [orig_arr, heatmap, overlay]
    cmaps  = ["gray", None, None]

    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")
        ax.set_facecolor("#0e1117")

    fig.suptitle(
        f"Prediction: {prediction}  |  Malignant Probability: {probability:.4f}",
        color=color, fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Standalone script
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print(f"Device     : {DEVICE}")
    print(f"Image      : {args.image}")
    print(f"Checkpoint : {CHECKPOINT}")

    # Load model
    print("\nLoading model...")
    model   = load_model()
    gradcam = GradCAM(model)

    # Preprocess
    tensor, orig_img = preprocess(args.image)

    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    cam, probability = gradcam.generate(tensor)

    prediction = "MALIGNANT" if probability > 0.5 else "BENIGN"
    print(f"\nPrediction  : {prediction}")
    print(f"Probability : {probability:.4f}")

    # Build figure
    fig = generate_gradcam_figure(orig_img, cam, probability)

    # Save
    out_path = args.save
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0e1117")
    plt.close(fig)
    print(f"\nGrad-CAM saved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Grad-CAM for a mammogram image"
    )
    p.add_argument("--image", type=str, default="sample_malignant.jpg",
                   help="Path to input mammogram image")
    p.add_argument("--save",  type=str, default="gradcam_result.png",
                   help="Output path for Grad-CAM image")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
