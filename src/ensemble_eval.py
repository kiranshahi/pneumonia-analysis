
import argparse
from pathlib import Path
import json

import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_fscore_support,
)

from tqdm import tqdm
import numpy as np

from .data import make_loaders
from .model import create_model
from .utils import load_checkpoint

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
    parser.add_argument("--spec-target", type=float, default=0.90, help="Specificity target")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_to_idx = make_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    ckpt_paths = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    models, metas = [], []
    for p in ckpt_paths:
        ckpt = load_checkpoint(p)
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

    # Imbalance-aware metrics
    if probs.shape[1] == 2:
        y_score = probs[:, 1]
        y_true = labels_all
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        thresholds = np.arange(0.0, 1.001, 0.001)

        best_spec = { "thr": None, "sens": -1, "spec": None, "prec": None, }
        best_f1 = { "thr": None, "sens": None, "spec": None, "prec": None, "f1": -1, }

        for t in thresholds:
            preds_bin = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds_bin).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds_bin, average="binary", zero_division=0
            )
            if spec >= args.spec_target and recall > best_spec["sens"]:
                best_spec.update({"thr": t, "sens": recall, "spec": spec, "prec": precision})
            if f1 > best_f1["f1"]:
                best_f1.update({"thr": t, "sens": recall, "spec": spec, "prec": precision, "f1": f1})
        print("Imbalance-aware Metrics:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        if best_spec["thr"] is not None:
            print(
                "  Specificity>="
                f"{args.spec_target:.2f}: thr={best_spec['thr']:.3f} sens={best_spec['sens']:.3f}"
                f" spec={best_spec['spec']:.3f} prec={best_spec['prec']:.3f}"
            )
        print(
            "  Best F1: "
            f"thr={best_f1['thr']:.3f} sens={best_f1['sens']:.3f} spec={best_f1['spec']:.3f} "
            f"prec={best_f1['prec']:.3f} f1={best_f1['f1']:.3f}"
        )
    else:
        roc_micro = roc_auc_score(labels_all, probs, multi_class="ovr", average="micro")
        roc_macro = roc_auc_score(labels_all, probs, multi_class="ovr", average="macro")
        pr_micro = average_precision_score(labels_all, probs, average="micro")
        pr_macro = average_precision_score(labels_all, probs, average="macro")
        
        print("Imbalance-aware Metrics:")
        print(f"  ROC-AUC micro={roc_micro:.4f} macro={roc_macro:.4f}")
        print(f"  PR-AUC  micro={pr_micro:.4f} macro={pr_macro:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(labels_all, preds))

    print("\nModels used:")
    for meta in metas:
        print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
