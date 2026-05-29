"""
evaluate.py
-----------
Full evaluation report for the trained DenseNet121 model.

Generates and saves to results/:
    - ROC curve
    - Precision-Recall curve
    - Confusion matrix heatmap
    - Threshold analysis (find optimal threshold for high malignant recall)
    - Training history plots
    - Classification report (text)
    - Summary CSV

Run:
    py -3.12 evaluate.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    roc_auc_score,
)

from dataset import get_loaders, MammogramDataset, get_transforms, encode_label
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Collect all predictions and true labels from a dataloader."""
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        images  = images.to(device, non_blocking=True)
        probs   = torch.softmax(model(images), dim=1)[:, 1]
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: ROC Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(y_true, y_probs, out_dir: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="crimson", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="crimson")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — DenseNet121 (Mass + Calc)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve          -> {path}")
    return roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curve(y_true, y_probs, out_dir: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.08, color="steelblue")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — DenseNet121", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "pr_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  PR curve           -> {path}")
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, threshold: float, out_dir: Path):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Benign", "Malignant"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.2f})",
                 fontsize=12, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix   -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Threshold Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_analysis(y_true, y_probs, out_dir: Path) -> float:
    """
    Plot precision, recall, F1, accuracy across thresholds.
    Find the lowest threshold where malignant recall >= 0.90.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s, accs = [], [], [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        acc  = (tp + tn) / len(y_true)
        precisions.append(prec); recalls.append(rec)
        f1s.append(f1);          accs.append(acc)

    # Find threshold for recall >= 0.90
    best_thresh = None
    for t, r in zip(thresholds, recalls):
        if r >= 0.90:
            best_thresh = t
            break

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, recalls,    color="red",    lw=2, label="Malignant Recall")
    ax.plot(thresholds, precisions, color="blue",   lw=2, label="Malignant Precision")
    ax.plot(thresholds, f1s,        color="green",  lw=2, label="F1 Score")
    ax.plot(thresholds, accs,       color="orange", lw=2, ls="--", label="Accuracy")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="Default t=0.50")
    if best_thresh is not None:
        ax.axvline(best_thresh, color="red", ls=":", lw=2,
                   label=f"Recall≥0.90 @ t={best_thresh:.2f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Analysis — DenseNet121", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = out_dir / "threshold_analysis.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Threshold analysis -> {path}")
    return best_thresh if best_thresh is not None else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Training History
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(hist_path: Path, out_dir: Path):
    if not hist_path.exists():
        print(f"  [!] History file not found: {hist_path}")
        return

    h = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History — DenseNet121 (Mass + Calc)",
                 fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(h["epoch"], h["tr_loss"], label="Train", color="steelblue")
    axes[0].plot(h["epoch"], h["vl_loss"], label="Val",   color="orange")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Epoch")

    # Accuracy
    axes[1].plot(h["epoch"], h["tr_acc"], label="Train", color="steelblue")
    axes[1].plot(h["epoch"], h["vl_acc"], label="Val",   color="orange")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch")

    # AUC
    axes[2].plot(h["epoch"], h["vl_auc"], color="crimson", lw=2, label="Val AUC")
    best_ep  = h.loc[h["vl_auc"].idxmax(), "epoch"]
    best_auc = h["vl_auc"].max()
    axes[2].axvline(best_ep, color="gray", ls="--", lw=1,
                    label=f"Best ep {best_ep} ({best_auc:.4f})")
    axes[2].set_title("Val AUC"); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    path = out_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training history   -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("checkpoints") / "best_combined_densenet121.pth"
    hist_path = Path("checkpoints") / "history_combined_densenet121.csv"
    out_dir   = Path("results")
    out_dir.mkdir(exist_ok=True)

    print(f"Device : {device}")

    # ── load model ────────────────────────────────────────────────────────
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run train.py first.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Best epoch : {ckpt['epoch']}  |  val_auc : {ckpt['val_auc']:.4f}")

    # ── load test data ────────────────────────────────────────────────────
    print("\nLoading test data...")
    _, _, test_loader = get_loaders(
        image_size  = 224,
        batch_size  = 32,
        num_workers = 2,
        image_types = ["crop", "full"],
    )

    # ── run inference ─────────────────────────────────────────────────────
    print("Running inference on test set...")
    y_true, y_probs = run_inference(model, test_loader, device)
    y_pred_05       = (y_probs >= 0.5).astype(int)

    # ── generate all plots ────────────────────────────────────────────────
    print(f"\nSaving evaluation outputs to: {out_dir}/\n")
    roc_auc_val = plot_roc(y_true, y_probs, out_dir)
    ap_val      = plot_pr_curve(y_true, y_probs, out_dir)
    best_thresh = plot_threshold_analysis(y_true, y_probs, out_dir)
    plot_confusion_matrix(y_true, y_pred_05, threshold=0.5, out_dir=out_dir)
    plot_training_history(hist_path, out_dir)

    # ── print classification report ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT  (threshold = 0.50)")
    print("=" * 60)
    acc_05 = (y_pred_05 == y_true).mean()
    print(f"\n  Test Accuracy : {acc_05:.4f}")
    print(f"  Test ROC-AUC  : {roc_auc_val:.4f}")
    print(f"  Avg Precision : {ap_val:.4f}\n")
    print(classification_report(
        y_true, y_pred_05,
        target_names=["Benign", "Malignant"],
        digits=4,
    ))

    # ── high-recall threshold report ──────────────────────────────────────
    if best_thresh != 0.5:
        y_pred_hr = (y_probs >= best_thresh).astype(int)
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REPORT  (threshold = {best_thresh:.2f}, recall ≥ 0.90)")
        print("=" * 60)
        print(classification_report(
            y_true, y_pred_hr,
            target_names=["Benign", "Malignant"],
            digits=4,
        ))
        plot_confusion_matrix(y_true, y_pred_hr,
                              threshold=best_thresh, out_dir=out_dir)

    # ── save summary ──────────────────────────────────────────────────────
    summary = pd.DataFrame([{
        "model":        "DenseNet121",
        "dataset":      "mass + calc (combined)",
        "image_types":  "crop + full",
        "best_epoch":   ckpt["epoch"],
        "val_auc":      round(ckpt["val_auc"], 4),
        "test_auc":     round(roc_auc_val, 4),
        "test_ap":      round(ap_val, 4),
        "test_acc":     round(float(acc_05), 4),
        "best_thresh":  round(best_thresh, 2),
    }])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")
    print(f"All plots     -> {out_dir}/")


if __name__ == "__main__":
    main()
