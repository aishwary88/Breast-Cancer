"""
Mammogram Preprocessing Pipeline
==================================
Maps patient_id from CSV files to their corresponding JPEG images in the
jpeg/ dataset by extracting the SeriesInstanceUID from the image file path
column, locating the matching folder, and copying all images with patient-
aware filenames.

Output naming convention:
  {patient_id}_{side}_{view}_{image_type}_{series_uid_short}_{filename}.jpg

Image types:
  - full   : full mammogram images
  - crop   : cropped images (ROI region)
  - mask   : ROI mask images

All renamed images are written to:  output_images/
A mapping CSV is written to:        output_images/image_mapping.csv
"""

import os
import re
import shutil
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
JPEG_ROOT   = Path("jpeg")
OUTPUT_DIR  = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILES = [
    Path("csv/mass_case_description_train_set.csv"),
    Path("csv/mass_case_description_test_set.csv"),
    Path("csv/calc_case_description_train_set.csv"),
    Path("csv/calc_case_description_test_set.csv"),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def extract_series_uid(file_path_str: str) -> str | None:
    """
    Extract the SeriesInstanceUID from a CBIS-DDSM image file path.

    The path format is:
      <StudyLabel>/<StudyInstanceUID>/<SeriesInstanceUID>/000000.dcm

    The jpeg/ folders are named by the SeriesInstanceUID, which is the
    SECOND UID segment in the path (the last UID before the filename).
    We collect all UID-like parts and return the last one found.
    """
    if not isinstance(file_path_str, str):
        return None
    parts = file_path_str.strip().split("/")
    uid_parts = [p for p in parts if re.match(r"^1\.3\.6\.", p)]
    # The SeriesInstanceUID is the last UID in the path
    return uid_parts[-1] if uid_parts else None


def find_jpeg_folder(series_uid: str) -> Path | None:
    """Return the jpeg/<series_uid> folder if it exists, else None."""
    if not series_uid:
        return None
    folder = JPEG_ROOT / series_uid
    return folder if folder.is_dir() else None


def get_image_type(col_name: str) -> str:
    """Map CSV column name to a short image-type label."""
    col = col_name.lower()
    if "cropped" in col:
        return "crop"
    if "roi" in col or "mask" in col:
        return "mask"
    return "full"


def safe_filename(value: str) -> str:
    """Strip characters that are unsafe in filenames."""
    return re.sub(r"[^\w\-]", "_", str(value))


# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────

def build_mapping(csv_files: list[Path]) -> pd.DataFrame:
    """
    Read all CSV files and build a unified mapping of:
      patient_id, side, view, image_type, series_uid, source_jpg, dest_name
    """
    records = []

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"  [SKIP] CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        # Normalise column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # Identify the three image-path columns
        path_cols = {
            "full": "image file path",
            "crop": "cropped image file path",
            "mask": "ROI mask file path",
        }

        for image_type, col in path_cols.items():
            if col not in df.columns:
                continue

            for _, row in df.iterrows():
                patient_id = str(row.get("patient_id", "")).strip()
                side       = str(row.get("left or right breast", "")).strip().upper()
                view       = str(row.get("image view", "")).strip().upper()
                raw_path   = str(row.get(col, "")).strip()

                if not patient_id or not raw_path or raw_path.lower() == "nan":
                    continue

                series_uid = extract_series_uid(raw_path)
                if not series_uid:
                    continue

                jpeg_folder = find_jpeg_folder(series_uid)
                if not jpeg_folder:
                    continue

                # Collect every .jpg in that folder
                jpg_files = sorted(jpeg_folder.glob("*.jpg"))
                if not jpg_files:
                    continue

                for jpg_path in jpg_files:
                    records.append({
                        "patient_id":  patient_id,
                        "side":        side,
                        "view":        view,
                        "image_type":  image_type,
                        "series_uid":  series_uid,
                        "source_path": str(jpg_path),
                        "source_file": jpg_path.name,
                    })

    mapping = pd.DataFrame(records).drop_duplicates(subset=["source_path"])
    return mapping


def generate_dest_names(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Add a dest_name column.  When multiple images share the same
    (patient_id, side, view, image_type) we append a counter so names
    stay unique.
    """
    mapping = mapping.copy()
    mapping["dest_name"] = ""

    # Count occurrences per group to decide whether to add a suffix
    group_counter: dict[str, int] = {}

    for idx, row in mapping.iterrows():
        base = (
            f"{safe_filename(row['patient_id'])}"
            f"_{safe_filename(row['side'])}"
            f"_{safe_filename(row['view'])}"
            f"_{row['image_type']}"
        )
        group_counter[base] = group_counter.get(base, 0) + 1
        count = group_counter[base]
        dest = f"{base}_{count:02d}.jpg"
        mapping.at[idx, "dest_name"] = dest

    return mapping


def copy_images(mapping: pd.DataFrame, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Copy source images to output_dir with the new dest_name.
    Returns a summary dict.
    """
    copied = skipped = errors = 0

    for _, row in mapping.iterrows():
        src  = Path(row["source_path"])
        dest = output_dir / row["dest_name"]

        if not src.exists():
            skipped += 1
            continue

        if dest.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] {src.name}  ->  {dest.name}")
            copied += 1
            continue

        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as exc:
            print(f"  [ERROR] {src} → {dest}: {exc}")
            errors += 1

    return {"copied": copied, "skipped": skipped, "errors": errors}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run(dry_run: bool = False):
    print("=" * 60)
    print("Mammogram Preprocessing Pipeline")
    print("=" * 60)

    print("\n[1/4] Building patient -> image mapping from CSV files ...")
    mapping = build_mapping(CSV_FILES)
    if mapping.empty:
        print("\n[!] No images could be mapped. Check that the jpeg/ folder "
              "exists and CSV paths are correct.")
        return

    print(f"      Found {len(mapping):,} unique source images across "
          f"{mapping['patient_id'].nunique():,} patients.")

    print("\n[2/4] Generating destination filenames ...")
    mapping = generate_dest_names(mapping)

    print("\n[3/4] Copying and renaming images ...")
    summary = copy_images(mapping, OUTPUT_DIR, dry_run=dry_run)
    action  = "Would copy" if dry_run else "Copied"
    print(f"      {action}:  {summary['copied']:,}")
    print(f"      Skipped: {summary['skipped']:,}  (already exist or source missing)")
    if summary["errors"]:
        print(f"      Errors:  {summary['errors']:,}")

    print("\n[4/4] Saving mapping CSV ...")
    mapping_csv = OUTPUT_DIR / "image_mapping.csv"
    mapping.to_csv(mapping_csv, index=False)
    print(f"      Saved -> {mapping_csv}")

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess CBIS-DDSM mammogram images: map patient IDs to images and rename."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying any files.",
    )
    args = parser.parse_args()

    run(dry_run=args.dry_run)
