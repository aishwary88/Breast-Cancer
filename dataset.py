"""
dataset.py
----------
Unified dataset loader for the CBIS-DDSM breast cancer classification project.

Combines mass_combined.csv + calc_combined.csv into a single dataframe,
uses image_mapping.csv to verify image existence, and builds PyTorch
DataLoaders ready for DenseNet121 (or any CNN).

Label encoding (binary classification):
    MALIGNANT               -> 1
    BENIGN                  -> 0
    BENIGN_WITHOUT_CALLBACK -> 0

Image types used:
    crop  - cropped ROI around the abnormality
    full  - full mammogram image
    (mask is excluded — not useful for classification)
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "MALIGNANT":               1,
    "BENIGN":                  0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

IMAGE_DIR   = Path("output_images")
MASS_CSV    = IMAGE_DIR / "mass_combined.csv"
CALC_CSV    = IMAGE_DIR / "calc_combined.csv"
MAPPING_CSV = IMAGE_DIR / "image_mapping.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Label helper
# ─────────────────────────────────────────────────────────────────────────────

def encode_label(pathology: str) -> int:
    """Map pathology string to binary label (0 = benign, 1 = malignant)."""
    return LABEL_MAP.get(str(pathology).strip().upper(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(image_size: int = 224, augment: bool = False) -> T.Compose:
    """
    Build the image transform pipeline.

    Training (augment=True):
        - Random crop after slight upscale
        - Horizontal + vertical flip
        - Random rotation ±15°
        - Brightness / contrast jitter
        - Normalize with ImageNet stats

    Validation / Test (augment=False):
        - Resize only
        - Normalize with ImageNet stats
    """
    # ImageNet normalization stats (work well for grayscale→RGB conversion)
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    if augment:
        return T.Compose([
            T.Grayscale(num_output_channels=3),                        # L → RGB
            T.Resize((int(image_size * 1.12), int(image_size * 1.12))),# slight upscale
            T.RandomCrop(image_size),                                  # random crop
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — combine mass + calc
# ─────────────────────────────────────────────────────────────────────────────

def load_combined_dataframe(
    image_dir:   Path = IMAGE_DIR,
    image_types: list = None,
) -> pd.DataFrame:
    """
    Load and merge mass_combined.csv + calc_combined.csv.

    Steps:
        1. Read both CSVs
        2. Add a 'source' column ('mass' or 'calc')
        3. Concatenate into one dataframe
        4. Keep only rows whose image file actually exists on disk
        5. Encode binary labels

    Returns a clean dataframe with columns:
        patient_id, split, side, view, image_type, dest_name,
        pathology, label, source, + all metadata columns
    """
    if image_types is None:
        image_types = ["crop", "full"]   # exclude mask

    # ── load mass ─────────────────────────────────────────────────────────
    mass = pd.read_csv(MASS_CSV)
    mass.columns = [c.strip() for c in mass.columns]
    mass["source"] = "mass"

    # ── load calc ─────────────────────────────────────────────────────────
    calc = pd.read_csv(CALC_CSV)
    calc.columns = [c.strip() for c in calc.columns]
    calc["source"] = "calc"

    # ── combine (outer join keeps all columns, fills missing with NaN) ────
    combined = pd.concat([mass, calc], ignore_index=True, sort=False)

    # ── filter image types ────────────────────────────────────────────────
    combined = combined[combined["image_type"].isin(image_types)].copy()

    # ── drop rows with missing pathology ──────────────────────────────────
    combined = combined.dropna(subset=["pathology"])

    # ── encode labels ─────────────────────────────────────────────────────
    combined["label"] = combined["pathology"].apply(encode_label)

    # ── verify image files exist on disk ──────────────────────────────────
    combined["image_path"] = combined["dest_name"].apply(
        lambda x: str(image_dir / x)
    )
    exists_mask = combined["image_path"].apply(lambda p: Path(p).exists())
    n_missing   = (~exists_mask).sum()
    if n_missing > 0:
        print(f"  [!] Skipping {n_missing} rows — image file not found on disk")
    combined = combined[exists_mask].reset_index(drop=True)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MammogramDataset(Dataset):
    """
    PyTorch Dataset for mammogram images.

    Parameters
    ----------
    records   : list of dicts, each with 'image_path' and 'label'
    transform : torchvision transform pipeline
    """

    def __init__(self, records: list[dict], transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Open as grayscale — transform will convert to 3-channel
        img   = Image.open(rec["image_path"]).convert("L")
        label = rec["label"]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    @property
    def class_counts(self) -> dict:
        """Return {0: n_benign, 1: n_malignant}."""
        labels = [r["label"] for r in self.records]
        return {c: labels.count(c) for c in sorted(set(labels))}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(
    image_size:  int   = 224,
    batch_size:  int   = 32,
    val_split:   float = 0.2,
    num_workers: int   = 2,
    seed:        int   = 42,
    image_types: list  = None,
    image_dir:   Path  = IMAGE_DIR,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the combined mass+calc dataset.

    - Train split uses augmentation + WeightedRandomSampler for class balance
    - Val / Test splits use no augmentation
    - Train/test split is taken from the 'split' column in the CSVs
    - Validation is carved from the training split

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    if image_types is None:
        image_types = ["crop", "full"]

    # ── load combined dataframe ───────────────────────────────────────────
    df = load_combined_dataframe(image_dir=image_dir, image_types=image_types)

    print(f"\nCombined dataset summary:")
    print(f"  Total images  : {len(df):,}")
    print(f"  Sources       : {df['source'].value_counts().to_dict()}")
    print(f"  Image types   : {df['image_type'].value_counts().to_dict()}")
    print(f"  Labels        : benign={int((df['label']==0).sum())}  "
          f"malignant={int((df['label']==1).sum())}")
    print(f"  Splits        : {df['split'].value_counts().to_dict()}")

    # ── separate train / test by original CSV split column ────────────────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # ── carve validation from train (stratified by label) ─────────────────
    random.seed(seed)
    np.random.seed(seed)

    # Stratified split: keep class ratio in val
    benign_idx    = train_df[train_df["label"] == 0].index.tolist()
    malignant_idx = train_df[train_df["label"] == 1].index.tolist()
    random.shuffle(benign_idx)
    random.shuffle(malignant_idx)

    n_val_b = int(len(benign_idx)    * val_split)
    n_val_m = int(len(malignant_idx) * val_split)

    val_idx   = benign_idx[:n_val_b]    + malignant_idx[:n_val_m]
    train_idx = benign_idx[n_val_b:]    + malignant_idx[n_val_m:]

    def make_records(subset_df: pd.DataFrame) -> list[dict]:
        """Convert dataframe rows to record dicts."""
        return [
            {
                "image_path": row["image_path"],
                "label":      int(row["label"]),
                "patient_id": str(row.get("patient_id", "")),
                "pathology":  str(row.get("pathology", "")),
                "source":     str(row.get("source", "")),
            }
            for _, row in subset_df.iterrows()
        ]

    tr_records  = make_records(train_df.loc[train_idx])
    val_records = make_records(train_df.loc[val_idx])
    te_records  = make_records(test_df)

    print(f"\n  Train : {len(tr_records):,} images")
    print(f"  Val   : {len(val_records):,} images")
    print(f"  Test  : {len(te_records):,} images")

    # ── transforms ────────────────────────────────────────────────────────
    train_tf = get_transforms(image_size, augment=True)
    eval_tf  = get_transforms(image_size, augment=False)

    train_ds = MammogramDataset(tr_records,  transform=train_tf)
    val_ds   = MammogramDataset(val_records, transform=eval_tf)
    test_ds  = MammogramDataset(te_records,  transform=eval_tf)

    # ── WeightedRandomSampler — balances class distribution per batch ──────
    labels      = [r["label"] for r in tr_records]
    counts      = np.bincount(labels)
    weights_cls = 1.0 / counts
    sample_wts  = torch.tensor([weights_cls[l] for l in labels], dtype=torch.float)
    sampler     = WeightedRandomSampler(
        weights     = sample_wts,
        num_samples = len(sample_wts),
        replacement = True,
    )

    cc = train_ds.class_counts
    print(f"  Train labels  : benign={cc.get(0,0)}  malignant={cc.get(1,0)}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    loader_kwargs = dict(
        num_workers       = num_workers,
        pin_memory        = torch.cuda.is_available(),
        persistent_workers= num_workers > 0,
        prefetch_factor   = 2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
