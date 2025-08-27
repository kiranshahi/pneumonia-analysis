
import argparse
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

from .data import make_loaders
from .model import create_model
from .utils import load_checkpoint

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, desc="Test", leave=False):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test split")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--spec-target", type=float, default=0.90, help="Specificity target")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_to_idx = make_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    ckpt = load_checkpoint(args.checkpoint)
    arch = ckpt.get("arch","resnet18")
    model = create_model(num_classes=len(class_to_idx), arch=arch, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    logits, labels = predict(model, loaders["test"], device)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    labels_np = labels.numpy()

    print("Classification report:")
    print(classification_report(labels_np, preds, target_names=[idx_to_class[i] for i in range(len(idx_to_class))]))

    # Imbalance-aware metrics
    if probs.shape[1] == 2:
        y_score = probs[:, 1]
        y_true = labels_np
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        thresholds = np.arange(0.0, 1.001, 0.001)
        best_spec = { "thr": None, "sens": -1, "spec": None, "prec": None}
        best_f1 = { "thr": None, "sens": None, "spec": None, "prec": None, "f1": -1}
        for t in thresholds:
            preds_bin = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds_bin).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds_bin, average="binary", zero_division=0)
            
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
        roc_micro = roc_auc_score(labels_np, probs, multi_class="ovr", average="micro")
        roc_macro = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
        pr_micro = average_precision_score(labels_np, probs, average="micro")
        pr_macro = average_precision_score(labels_np, probs, average="macro")
        print("Imbalance-aware Metrics:")
        print(f"  ROC-AUC micro={roc_micro:.4f} macro={roc_macro:.4f}")
        print(f"  PR-AUC  micro={pr_micro:.4f} macro={pr_macro:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(labels_np, preds))

if __name__ == "__main__":
    main()
