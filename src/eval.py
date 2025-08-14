
import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm

from .data import make_loaders
from .model import create_model

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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_to_idx = make_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    if probs.shape[1] == 2:
        auc = roc_auc_score(labels_np, probs[:,1])
        print(f"ROC AUC: {auc:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(labels_np, preds))

if __name__ == "__main__":
    main()
