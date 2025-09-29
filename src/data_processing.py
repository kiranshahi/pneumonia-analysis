import csv
from pathlib import Path
import os
import re
import shutil
import pandas as pd
import numpy as np


def export_filenames(src_root, out_csv="all_images.csv"):
    """
    Export all JPEG file paths and class labels to a CSV.
    Expects directory structure:
        src_root/
          Normal/
          Pneumonia/
    """
    src = Path(src_root)
    rows = []

    for cls in ["Normal", "Pneumonia"]:
        cls_dir = src / cls
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found, skipping.")
            continue
        for fname in cls_dir.rglob("*.jpeg"):
            if fname.is_file():
                rows.append([str(fname.resolve()), cls])

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {out_csv}")

# export_filenames("D:\\Learn\\pneumonia_detection\\data\\chest_xray", "all_images.csv")



# ====== CONFIG ======
CSV_PATH   = "all_images.csv"       # your merged CSV (cols: image_path, label, etc.)
OUTPUT_DIR = "D:\\Learn\\pneumonia_detection\\data\\chest_xray"           # where to save split CSVs
IMAGE_ROOT = "D:\\Learn\\pneumonia_detection\\data\\chest_xray"           # root dir where images actually live (for copying)
MAKE_FOLDERS_AND_COPY = True                # set True to copy files into train/val/test/<label>/

# ====== HELPERS ======

# Extract patient_id from filename/path
# - Matches patient Id
# - If no match, falls back to whole stem.
PATTERN_PNEUMONIA = re.compile(r"^(person\d+)", re.IGNORECASE)
PATTERN_NORMAL = re.compile(r"^([A-Za-z]+-\d{4}-\d{4})", re.IGNORECASE)

def get_stem(p):
    name = os.path.basename(str(p))
    stem, _ = os.path.splitext(name)
    return stem

def infer_patient_id(path_or_name: str) -> str:
    stem = get_stem(path_or_name)
    m = PATTERN_PNEUMONIA.match(stem)
    if m:
        return m.group(1).upper()
    m = PATTERN_NORMAL.match(stem)
    if m:
        return m.group(1).upper()
    parts = stem.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3]).upper()
    return stem.upper()

def normalize_label(x):
    s = str(x).strip().lower()
    if s in {"normal", "norm", "0"}:
        return "NORMAL"
    if s in {"pneumonia", "pneu", "1"}:
        return "PNEUMONIA"
    return str(x).strip().upper()

def ensure_min_one_per_class(patients_by_label, val_set, test_set, train_set, rng):
    """
    Ensure val and test each have at least one patient from each label, if available.
    If missing, move one suitable patient from train.
    """
    for lab, pids in patients_by_label.items():
        # val
        if not any(pid in val_set for pid in pids) and any(pid in train_set for pid in pids):
            # move one from train -> val
            candidates = [pid for pid in pids if pid in train_set]
            if candidates:
                chosen = rng.choice(candidates)
                train_set.remove(chosen)
                val_set.add(chosen)
        # test
        if not any(pid in test_set for pid in pids) and any(pid in train_set for pid in pids):
            # move one from train -> test
            candidates = [pid for pid in pids if pid in train_set]
            if candidates:
                chosen = rng.choice(candidates)
                train_set.remove(chosen)
                test_set.add(chosen)

def stratified_group_split(patient_df, label_col="patient_label", train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    train_set, val_set, test_set = set(), set(), set()

    # Split per label to keep proportions
    patients_by_label = {}
    for lab, sub in patient_df.groupby(label_col):
        pids = sub["patient_id"].tolist()
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(train_ratio * n))
        n_val   = int(round(val_ratio * n))
        n_test  = n - n_train - n_val

        # tiny-class guards
        if n_train == 0 and n > 0:
            n_train = 1
            n_test = max(0, n - n_train - n_val)  # re-balance residual

        train_set.update(pids[:n_train])
        val_set.update(pids[n_train:n_train+n_val])
        test_set.update(pids[n_train+n_val:])

        patients_by_label[lab] = set(pids)

    # Make sure val/test each have at least one of every available label
    ensure_min_one_per_class(patients_by_label, val_set, test_set, train_set, rng)

    # Sanity: ensure sets are disjoint
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    return train_set, val_set, test_set

def materialize_folders(df, path_col, label_col, out_base):
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        for lab in sdf[label_col].dropna().unique():
            os.makedirs(os.path.join(out_base, split, str(lab)), exist_ok=True)
        for _, row in sdf.iterrows():
            src = row[path_col]
            if pd.isna(src):
                continue
            # If CSV stores only basenames, join with IMAGE_ROOT
            if not os.path.isabs(src):
                src = os.path.join(IMAGE_ROOT, src)
            if not os.path.exists(src):
                # Try fallback: join basename with IMAGE_ROOT
                alt = os.path.join(IMAGE_ROOT, os.path.basename(str(src)))
                if os.path.exists(alt):
                    src = alt
                else:
                    print(f"⚠️ Missing file: {src}")
                    continue
            dst = os.path.join(out_base, split, str(row[label_col]), os.path.basename(str(src)))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

# ====== MAIN ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Detect likely path column and label column
path_col = None
for c in df.columns:
    if any(k in c.lower() for k in ["path", "file", "image"]):
        path_col = c
        break
if path_col is None:
    raise ValueError("Couldn't detect an image path column. Add a column like 'image_path'.")

label_col = None
for c in df.columns:
    if c.lower() in {"label", "class", "category", "diagnosis", "target", "status", "y"}:
        label_col = c
        break
if label_col is None:
    raise ValueError("Couldn't detect a label column. Add a 'label' (NORMAL/PNEUMONIA).")

# Normalize labels
df[label_col] = df[label_col].apply(normalize_label)

# Build/overwrite patient_id using the RSNA/Kermany-style pattern
df["patient_id"] = df[path_col].apply(infer_patient_id)

# Patient-level table (use the mode label per patient)
patient_df = (
    df.groupby("patient_id")[label_col]
      .agg(lambda s: s.value_counts().index[0])
      .reset_index()
      .rename(columns={label_col: "patient_label"})
)

# Perform grouped split
train_p, val_p, test_p = stratified_group_split(patient_df, label_col="patient_label", train_ratio=0.70, val_ratio=0.15, seed=42)

def assign_split(pid):
    if pid in train_p: return "train"
    if pid in val_p:   return "val"
    if pid in test_p:  return "test"
    return "unassigned"

df["split"] = df["patient_id"].apply(assign_split)

# Save CSVs
train_csv = os.path.join(OUTPUT_DIR, "train_split.csv")
val_csv   = os.path.join(OUTPUT_DIR, "val_split.csv")
test_csv  = os.path.join(OUTPUT_DIR, "test_split.csv")
all_csv   = os.path.join(OUTPUT_DIR, "all_splits.csv")

df.to_csv(all_csv, index=False)
df[df["split"]=="train"].to_csv(train_csv, index=False)
df[df["split"]=="val"].to_csv(val_csv, index=False)
df[df["split"]=="test"].to_csv(test_csv, index=False)

print("Saved:")
print(" -", train_csv)
print(" -", val_csv)
print(" -", test_csv)
print(" -", all_csv)

# Quick check: unique patients per split & class
patient_counts = (
    df.groupby(["split","patient_id",label_col])
      .size().reset_index().groupby(["split", label_col])["patient_id"]
      .nunique().unstack(fill_value=0)
)
print("\nUnique patients per split (by label):\n", patient_counts)

# Optional: copy images into train/val/test/<label> folders
if MAKE_FOLDERS_AND_COPY:
    out_base = os.path.join(OUTPUT_DIR, "dataset")
    print("\nCopying images into:", out_base)
    materialize_folders(df, path_col=path_col, label_col=label_col, out_base=out_base)
    print("Done.")

