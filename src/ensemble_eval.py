
import argparse
from pathlib import Path
import json

import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm
import numpy as np

from .data import make_loaders
from .model import create_model

@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_logits = []
    for images, _ in tqdm(loader, desc="Predict", leave=False):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits)

def main():
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of checkpoints")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated list of .pt files")
    parser.add_argument("--weights", type=str, default="uniform", choices=["uniform","auto_auc"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_to_idx = make_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    ckpt_paths = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    models, metas = [], []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu")
        arch = ckpt.get("arch","resnet18")
        model = create_model(num_classes=len(class_to_idx), arch=arch, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        models.append(model)
        metas.append({"path": p, "val_auc": ckpt.get("val_auc", None), "val_acc": ckpt.get("val_acc", None), "arch": arch})

    # Iterate test loader once per model (simple & memory friendly)
    logits_models = []
    labels_all = None
    for m in models:
        # predict stores no labels, we will get labels separately
        logits = predict_logits(m, loaders["test"], device)
        logits_models.append(logits.numpy())
    # Get labels once
    lbls = []
    for _, labels in loaders["test"]:
        lbls.append(labels)
    labels_all = torch.cat(lbls).numpy()

    logits_models = np.stack(logits_models, axis=0)  # [M, N, C]

    # Weights
    if args.weights == "uniform":
        w = np.ones((logits_models.shape[0], 1, 1), dtype=np.float32)
    else:
        aucs = [meta["val_auc"] if meta["val_auc"] is not None else 0.5 for meta in metas]
        w = np.array(aucs, dtype=np.float32).reshape(-1,1,1)
        s = np.sum(w)
        if s <= 0:
            w = np.ones_like(w)
        else:
            w = w / s

    ensemble_logits = np.sum(w * logits_models, axis=0)  # [N, C]
    probs = (torch.softmax(torch.from_numpy(ensemble_logits), dim=1).numpy())
    preds = probs.argmax(axis=1)

    print("Classification report (ensemble):")
    print(classification_report(labels_all, preds, target_names=[idx_to_class[i] for i in range(len(idx_to_class))]))

    if probs.shape[1] == 2:
        auc = roc_auc_score(labels_all, probs[:,1])
        print(f"ROC AUC (ensemble): {auc:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(labels_all, preds))

    print("\nModels used:")
    for meta in metas:
        print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
