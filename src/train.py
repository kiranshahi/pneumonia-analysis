
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from .data import make_loaders
from .model import create_model
from .utils import set_seed

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
    auc = None
    if logits_all.shape[1] == 2:
        probs = torch.softmax(logits_all, dim=1).numpy()[:,1]
        auc = roc_auc_score(labels_all.numpy(), probs)
    return running_loss / n, acc, auc

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
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18","densenet121","efficientnet_b0"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    loaders, class_to_idx = make_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size, seed=args.seed)
    model = create_model(num_classes=len(class_to_idx), arch=args.arch, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_path = str(Path(args.out_dir) / f"best_{args.arch}.pt")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], device, criterion, optimizer)
        va_loss, va_acc, va_auc = _eval_collect(model, loaders["val"], device, criterion)
        history["train"].append({"epoch": epoch, "loss": tr_loss, "acc": tr_acc})
        history["val"].append({"epoch": epoch, "loss": va_loss, "acc": va_acc, "auc": va_auc})
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc if va_auc is not None else 'NA'}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "arch": args.arch,
                "val_acc": va_acc,
                "val_auc": va_auc
            }, best_path)

    with open(Path(args.out_dir)/f"history_{args.arch}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val acc: {best_val_acc:.4f}. Saved to {best_path}")

if __name__ == "__main__":
    main()
