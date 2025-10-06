from pathlib import Path
import os, re, shutil, csv
import pandas as pd
import numpy as np


def export_filenames(src_root, out_csv="all_images.csv"):
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

OUTPUT_DIR = "D:\\Learn\\pneumonia_detection\\data\\chest_xray"           # where to save split CSVs
IMAGE_ROOT = "D:\\Learn\\pneumonia_detection\\data\\chest_xray"           # root dir where images actually live (for copying)
MAKE_FOLDERS_AND_COPY = True                # set True to copy files into train/val/test/<label>/

# Matches patient Id
PATTERN_PNEUMONIA = re.compile(r"^(person\d+)", re.IGNORECASE)
PATTERN_NORMAL = re.compile(r"^([A-Za-z]+-\d{4}-\d{4})", re.IGNORECASE)

def get_stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]

def infer_patient_id(path_or_name: str) -> str:
    stem = get_stem(path_or_name)
    m = PATTERN_PNEUMONIA.match(stem)
    if m: return m.group(1).upper()
    m = PATTERN_NORMAL.match(stem)
    if m: return m.group(1).upper()
    parts = stem.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3]).upper()
    return stem.upper()

def normalize_label(x):
    s = str(x).strip().lower()
    if s.startswith("norm"): return "NORMAL"
    if s.startswith("pneu"): return "PNEUMONIA"
    return s.upper()

def stratified_group_split(patient_df, label_col="patient_label", train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    train_set, val_set, test_set = set(), set(), set()
    patients_by_label = {}

    for lab, sub in patient_df.groupby(label_col):
        pids = sub["patient_id"].tolist()
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(train_ratio * n))
        n_val   = int(round(val_ratio * n))
        n_test  = n - n_train - n_val

        train_set.update(pids[:n_train])
        val_set.update(pids[n_train:n_train+n_val])
        test_set.update(pids[n_train+n_val:])

        patients_by_label[lab] = set(pids)

    return train_set, val_set, test_set

def materialize_folders(df, out_base):
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        for lab in sdf["label"].unique():
            os.makedirs(os.path.join(out_base, split, lab), exist_ok=True)
        for _, row in sdf.iterrows():
            src = row["path"]
            dst = os.path.join(out_base, split, row["label"], os.path.basename(str(src)))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

# ====== MAIN ======
rows = []
for cls in ["Normal", "Pneumonia"]:
    cls_dir = Path(IMAGE_ROOT) / cls
    for fname in cls_dir.rglob("*.jpeg"):
        rows.append({"path": str(fname),
                     "label": normalize_label(cls),
                     "patient_id": infer_patient_id(fname)})

df = pd.DataFrame(rows)

# Patient-level labels
patient_df = (
    df.groupby("patient_id")["label"]
      .agg(lambda s: s.value_counts().index[0])
      .reset_index()
      .rename(columns={"label":"patient_label"})
)

# Split
train_p, val_p, test_p = stratified_group_split(patient_df)

def assign_split(pid):
    if pid in train_p: return "train"
    if pid in val_p:   return "val"
    if pid in test_p:  return "test"
    return "unassigned"

df["split"] = df["patient_id"].apply(assign_split)

print(df["split"].value_counts())

if MAKE_FOLDERS_AND_COPY:
    out_base = os.path.join(OUTPUT_DIR, "dataset")
    materialize_folders(df, out_base)
    print("Dataset copied to:", out_base)