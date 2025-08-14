
# Pneumonia CNN (multi-model/ensemble) — Colab-friendly Project

Train, evaluate, and visualize Grad-CAM for binary chest X-ray classification (Normal vs Pneumonia) with **consistent preprocessing**, **patient-wise split**, and **multi-model ensembling**.

## Project layout
```
pneumonia_cnn_ensemble_project/
  pneumonia_cnn/
    __init__.py
    data.py
    model.py
    train.py
    eval.py
    gradcam.py
    train_many.py
    ensemble_eval.py
    utils.py
  requirements.txt
  README.md
```

## Colab quickstart

1) Upload/clone this project into Colab.

2) Install deps:
```bash
%cd /content/pneumonia_cnn_ensemble_project
!pip install -r requirements.txt
```

3) Mount Drive and set dataset path (folder containing `Normal/` and `Pneumonia/`):
```python
from google.colab import drive
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/Chest_xray'  # adjust
```

### Train single model (choose arch)
```bash
python -m pneumonia_cnn.train --data_dir "$data_dir" --out_dir "/content/drive/MyDrive/pneumonia_runs/resnet18" --epochs 5 --arch resnet18
python -m pneumonia_cnn.train --data_dir "$data_dir" --out_dir "/content/drive/MyDrive/pneumonia_runs/densenet121" --epochs 5 --arch densenet121
python -m pneumonia_cnn.train --data_dir "$data_dir" --out_dir "/content/drive/MyDrive/pneumonia_runs/efficientnet_b0" --epochs 5 --arch efficientnet_b0
```

### Train many in one go
```bash
python -m pneumonia_cnn.train_many --data_dir "$data_dir" --out_root "/content/drive/MyDrive/pneumonia_runs/archs" --epochs 5
```

### Evaluate a single model
```bash
python -m pneumonia_cnn.eval --data_dir "$data_dir" --checkpoint "/content/drive/MyDrive/pneumonia_runs/resnet18/best_resnet18.pt"
```

### Evaluate an ensemble
Uniform average:
```bash
python -m pneumonia_cnn.ensemble_eval --data_dir "$data_dir"   --checkpoints "/content/drive/MyDrive/pneumonia_runs/archs/resnet18/best_resnet18.pt,/content/drive/MyDrive/pneumonia_runs/archs/densenet121/best_densenet121.pt,/content/drive/MyDrive/pneumonia_runs/archs/efficientnet_b0/best_efficientnet_b0.pt"   --weights uniform
```
AUC-weighted average (uses validation AUC stored in each checkpoint):
```bash
python -m pneumonia_cnn.ensemble_eval --data_dir "$data_dir"   --checkpoints "/content/drive/MyDrive/pneumonia_runs/archs/resnet18/best_resnet18.pt,/content/drive/MyDrive/pneumonia_runs/archs/densenet121/best_densenet121.pt,/content/drive/MyDrive/pneumonia_runs/archs/efficientnet_b0/best_efficientnet_b0.pt"   --weights auto_auc
```

### Grad-CAM on any trained checkpoint
```bash
python -m pneumonia_cnn.gradcam --checkpoint "/content/drive/MyDrive/pneumonia_runs/resnet18/best_resnet18.pt" --image_path "/content/drive/MyDrive/Chest_xray/Pneumonia/person1413_virus_2423.jpg" --out_path "/content/gradcam_resnet18.png"
```

**Notes**
- Uses **ImageNet normalization** consistently across train/eval/Grad-CAM.
- Patient-wise split inferred from filename prefix (e.g., `person1234_*`). Adjust `infer_patient_id` if needed.
- Checkpoints store `arch`, `val_acc`, `val_auc` for later AUC-weighted ensembling.
