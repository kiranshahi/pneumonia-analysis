
from pathlib import Path
from typing import Dict
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import IMAGENET_MEAN, IMAGENET_STD

__all__ = [
    "default_transform",
    "infer_patient_id",
    "split_by_patient",
    "make_loaders",
    "compute_class_counts",
    "make_sample_weights_from_counts",
]

def default_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def infer_patient_id(name: str) -> str:
    base = Path(name).name
    token = base.split('_')[0]  # e.g., person1234
    return token

def split_by_patient(dataset: datasets.ImageFolder, ratios=(0.7, 0.15, 0.15), seed: int = 42):
    patient_to_indices = {}
    for idx, (path, _) in enumerate(dataset.samples):
        pid = infer_patient_id(path)
        patient_to_indices.setdefault(pid, []).append(idx)

    patients = list(patient_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(ratios[0]*n)
    n_val = int(ratios[1]*n)
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train+n_val])
    test_patients = set(patients[n_train+n_val:])

    def gather(pids):
        idxs = []
        for p in pids:
            idxs.extend(patient_to_indices[p])
        return Subset(dataset, idxs)

    return {
        "train": gather(train_patients),
        "val": gather(val_patients),
        "test": gather(test_patients),
    }

def make_loaders(root_dir: str, batch_size: int = 32, num_workers: int = 2, img_size: int = 224, seed: int = 42):
    transform = default_transform(img_size)
    ds = datasets.ImageFolder(root=root_dir, transform=transform)
    splits = split_by_patient(ds, seed=seed)
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k=='train'), num_workers=num_workers, pin_memory=True)
        for k, v in splits.items()
    }
    class_to_idx = ds.class_to_idx
    return loaders, class_to_idx

def _iter_labels(dataset):
    """Yield labels from dataset, supporting ``torch.utils.data.Subset``."""
    if isinstance(dataset, Subset):
        for idx in dataset.indices:
            # Access underlying dataset directly to avoid double wrapping
            _, label = dataset.dataset[idx]
            yield label
    else:
        for _, label in dataset:
            yield label


def compute_class_counts(dataset):
    """Compute the number of samples per class in ``dataset``.

    Handles ``torch.utils.data.Subset`` instances and returns a tensor of
    shape ``[num_classes]`` with the counts for each class.
    """
    # Determine number of classes from the underlying dataset
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    num_classes = len(getattr(base, "classes"))
    counts = torch.zeros(num_classes, dtype=torch.long)
    for label in _iter_labels(dataset):
        counts[label] += 1
    return counts


def make_sample_weights_from_counts(dataset, class_counts):
    """Generate per-sample weights inversely proportional to ``class_counts``.

    Each sample receives a weight of ``1 / class_counts[label]``.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float)
    weights = [1.0 / counts[label].item() for label in _iter_labels(dataset)]
    return torch.tensor(weights, dtype=torch.float)