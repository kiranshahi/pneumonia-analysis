
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

from .data import (
    make_loaders,
    compute_class_counts,
    make_sample_weights_from_counts
)
from .model import create_model
from .utils import (
    set_seed,
    class_weights_from_counts,
    FocalLoss,
    focal_alpha_from_counts,
    )

def _eval_collect(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    logits_all, labels_all = [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    n = len(loader.dataset)
    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    acc = (logits_all.argmax(1) == labels_all).float().mean().item()
    auc = pr_auc = None
    
    probs = torch.softmax(logits_all, dim=1).numpy()
    labels_np = labels_all.numpy()

    if probs.shape[1] == 2:
        auc = roc_auc_score(labels_np, probs[:, 1])
        pr_auc = average_precision_score(labels_np, probs[:, 1])
    else:
        labels_bin = label_binarize(labels_np, classes=np.arange(probs.shape[1]))
        auc = {
            "micro": roc_auc_score(labels_np, probs, multi_class="ovr", average="micro"),
            "macro": roc_auc_score(labels_np, probs, multi_class="ovr", average="macro"),
        }
        pr_auc = {
            "micro": average_precision_score(labels_bin, probs, average="micro"),
            "macro": average_precision_score(labels_bin, probs, average="macro"),
        }
    return running_loss / n, acc, auc, pr_auc

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += (logits.argmax(1) == labels).float().sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

def main():
    parser = argparse.ArgumentParser(description="Train CNN for Pneumonia classification (multi-arch)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18","densenet121","efficientnet_b0","resnet50","mobilenet_v2","vit_b_16"])
    parser.add_argument("--balance", type=str, choices=["none", "sampler", "class_weights"], default="none")
    parser.add_argument("--loss", type=str, choices=["ce", "ce_weighted", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--alpha-mode", type=str, choices=["none", "inv_freq"], default="none")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    loaders, class_to_idx = make_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size, seed=args.seed)
    
    class_counts = compute_class_counts(loaders["train"].dataset)
    print(f"Class counts: {class_counts.tolist()}")
    
    if args.balance == "sampler":
        sample_weights = make_sample_weights_from_counts(
            loaders["train"].dataset, class_counts
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        loaders["train"] = DataLoader(
            loaders["train"].dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=loaders["train"].num_workers,
            pin_memory=True,
        )

    model = create_model(
        num_classes=len(class_to_idx),
        arch=args.arch,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = None
    alpha = None
    if args.loss == "focal":
        alpha = (
            focal_alpha_from_counts(class_counts)
            if args.alpha_mode == "inv_freq"
            else None
        )
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha).to(device)
    elif args.loss == "ce_weighted" or args.balance == "class_weights":
        class_weights = class_weights_from_counts(class_counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()


    model = create_model(num_classes=len(class_to_idx), arch=args.arch, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    imbalance_summary = {
        "balance": args.balance,
        "loss": args.loss,
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "focal_alpha": alpha.tolist() if alpha is not None else None,
    }
    print(f"Imbalance summary: {imbalance_summary}")

    best_val_acc = 0.0
    best_path = str(Path(args.out_dir) / f"best_{args.arch}.pt")
    history = {"train": [], "val": [], "imbalance": imbalance_summary}

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], device, criterion, optimizer)
        va_loss, va_acc, va_auc, va_pr_auc = _eval_collect(model, loaders["val"], device, criterion)
        history["train"].append({"epoch": epoch, "loss": tr_loss, "acc": tr_acc})
        history["val"].append({"epoch": epoch, "loss": va_loss, "acc": va_acc, "auc": va_auc, "pr_auc": va_pr_auc})

        if isinstance(va_auc, dict):
            auc_str = f"micro={va_auc['micro']:.4f} macro={va_auc['macro']:.4f}"
        else:
            auc_str = f"{va_auc:.4f}" if va_auc is not None else "NA"

        if isinstance(va_pr_auc, dict):
            pr_auc_str = f"micro={va_pr_auc['micro']:.4f} macro={va_pr_auc['macro']:.4f}"
        else:
            pr_auc_str = f"{va_pr_auc:.4f}" if va_pr_auc is not None else "NA"

        print(
            f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
            f"roc_auc {auc_str} pr_auc {pr_auc_str}"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "arch": args.arch,
                "val_acc": va_acc,
                "val_auc": va_auc,
                "val_pr_auc": va_pr_auc,
            }, best_path)

    with open(Path(args.out_dir)/f"history_{args.arch}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val acc: {best_val_acc:.4f}. Saved to {best_path}")

if __name__ == "__main__":
    main()
