"""
train.py
--------
Train a unified DenseNet121 on the combined MASS + CALC dataset.

Run:
    py -3.12 train.py              # start fresh
    py -3.12 train.py --resume     # continue from last checkpoint

Saves:
    checkpoints/best_combined_densenet121.pth    <- best val AUC model
    checkpoints/resume_combined_densenet121.pth  <- full state for resuming
    checkpoints/history_combined_densenet121.csv <- per-epoch metrics
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure stdout is flushed immediately (important for log visibility)
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from dataset import get_loaders
from model import build_model, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Training loop — one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataloader.
    Returns (avg_loss, accuracy, roc_auc, y_true, y_probs).
    """
    model.eval()
    total_loss = correct = total = 0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)
        probs   = torch.softmax(outputs, dim=1)[:, 1]   # malignant probability

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    y_true   = np.array(all_labels)
    y_probs  = np.array(all_probs)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = float("nan")

    return avg_loss, accuracy, auc, y_true, y_probs


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── device setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── load combined dataset (mass + calc, crop + full) ──────────────────
    print("\n" + "=" * 60)
    print("Loading combined MASS + CALC dataset")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_loaders(
        image_size  = args.image_size,
        batch_size  = args.batch_size,
        val_split   = 0.2,
        num_workers = 2,
        image_types = ["crop", "full"],   # all useful image types
    )

    # ── build DenseNet121 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building DenseNet121 (pretrained on ImageNet)")
    print("=" * 60)

    model = build_model(
        num_classes     = 2,
        pretrained      = True,
        freeze_backbone = False,   # fine-tune the full network
        dropout         = 0.4,
    )
    model = model.to(device)

    params = count_parameters(model)
    print(f"  Total params     : {params['total']:,}")
    print(f"  Trainable params : {params['trainable']:,}")

    # ── loss, optimizer, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # Reduce LR by 0.5 when val AUC stops improving for 3 epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode    = "max",
        factor  = 0.5,
        patience= 3,
    )

    # ── checkpoint paths ──────────────────────────────────────────────────
    ckpt_dir    = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_path   = ckpt_dir / "best_combined_densenet121.pth"
    resume_path = ckpt_dir / "resume_combined_densenet121.pth"
    hist_path   = ckpt_dir / "history_combined_densenet121.csv"

    # ── resume from checkpoint if requested ───────────────────────────────
    start_epoch    = 1
    best_val_auc   = 0.0
    patience_count = 0
    history        = []

    if args.resume and resume_path.exists():
        print(f"\nResuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_auc   = ckpt["best_val_auc"]
        patience_count = ckpt["patience_count"]
        history        = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}  |  best_val_auc = {best_val_auc:.4f}")
    elif args.resume:
        print("\n[!] No resume checkpoint found — starting fresh.")

    # ── training loop ─────────────────────────────────────────────────────
    end_epoch = start_epoch + args.epochs - 1
    print(f"\n{'='*60}")
    print(f"Training  epochs {start_epoch} → {end_epoch}  "
          f"|  patience = {args.patience}")
    print(f"{'='*60}")
    print(f"\n{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VlLoss':>8}  {'VlAcc':>7}  {'VlAUC':>7}  "
          f"{'LR':>9}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()

        # train
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        # validate
        vl_loss, vl_acc, vl_auc, _, _ = evaluate(
            model, val_loader, criterion, device)

        # update LR scheduler based on val AUC
        scheduler.step(vl_auc)

        elapsed = time.time() - t0
        lr      = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_auc:>7.4f}  "
              f"{lr:>9.6f}  {elapsed:>5.0f}s")

        # record history
        history.append({
            "epoch":   epoch,
            "tr_loss": tr_loss, "tr_acc": tr_acc,
            "vl_loss": vl_loss, "vl_acc": vl_acc, "vl_auc": vl_auc,
        })

        # ── always save resume checkpoint (full state) ────────────────────
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_auc":    best_val_auc,
            "patience_count":  patience_count,
            "history":         history,
        }, resume_path)

        # ── save best model ───────────────────────────────────────────────
        if vl_auc > best_val_auc:
            best_val_auc   = vl_auc
            patience_count = 0
            torch.save({
                "epoch":     epoch,
                "model_state": model.state_dict(),
                "val_auc":   vl_auc,
                "val_acc":   vl_acc,
            }, best_path)
            print(f"     ** best model saved  (val_auc = {vl_auc:.4f})")
        else:
            patience_count += 1
            print(f"     no improvement  ({patience_count}/{args.patience})")
            if patience_count >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # ── save training history CSV ──────────────────────────────────────────
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"\nTraining history saved -> {hist_path}")

    # ── final test evaluation ─────────────────────────────────────────────
    print(f"\nLoading best model: {best_path}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Best epoch : {ckpt['epoch']}  |  val_auc : {ckpt['val_auc']:.4f}")

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    _, test_acc, test_auc, y_true, y_probs = evaluate(
        model, test_loader, criterion, device)
    y_pred = (y_probs >= 0.5).astype(int)

    print(f"\n  Accuracy : {test_acc:.4f}")
    print(f"  ROC-AUC  : {test_auc:.4f}\n")
    print(classification_report(
        y_true, y_pred,
        target_names=["Benign", "Malignant"],
        digits=4,
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows = actual, cols = predicted):")
    print(f"               Benign  Malignant")
    print(f"  Benign       {cm[0][0]:>6}  {cm[0][1]:>9}")
    print(f"  Malignant    {cm[1][0]:>6}  {cm[1][1]:>9}")
    print(f"\nBest model -> {best_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train unified DenseNet121 on combined MASS + CALC dataset"
    )
    p.add_argument("--image_size",   type=int,   default=224,
                   help="Input image size (default: 224)")
    p.add_argument("--epochs",       type=int,   default=25,
                   help="Max training epochs (default: 25)")
    p.add_argument("--batch_size",   type=int,   default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Adam weight decay (default: 1e-4)")
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early stopping patience (default: 5)")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from resume_combined_densenet121.pth")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # required on Windows
    args = parse_args()
    train(args)
