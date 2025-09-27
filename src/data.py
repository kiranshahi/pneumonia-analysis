
from pathlib import Path
from typing import Dict, Callable, Optional
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import IMAGENET_MEAN, IMAGENET_STD

AUG_POLICIES = {
    "light": dict(rotate=0.5, shift_scale=0.5, brightness_contrast=0.5, clahe=0.3, gamma=0.3, noise=0.2, motion_blur=0.1, median_blur=0.1, dropout=0.1)
}

class AlbumentationsTransform:
    """Convert PIL image > repeated grayscale > Albumentations → tensor."""
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        img = np.array(img.convert("L"))          # grayscale
        img = np.stack([img]*3, axis=-1)          # repeat channels
        return self.aug(image=img)["image"]
    
class TransformDataset(torch.utils.data.Dataset):
    """Apply a transform to an existing dataset or subset without altering it."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        try:
            base = dataset.dataset
        except AttributeError:
            base = dataset
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label

class RemapTargetsDataset(Dataset):
    """Wrap a dataset and remap class indices to a canonical mapping."""

    def __init__(self, dataset: Dataset, canonical: Dict[str, int]):
        self.dataset = dataset
        self._canonical = dict(canonical)

        try:
            base = dataset.dataset
        except AttributeError:
            base = dataset
        if not hasattr(base, "classes"):
            raise AttributeError("Dataset must expose a 'classes' attribute for remapping.")
        self._base_classes = base.classes

        # Provide ``classes`` / ``class_to_idx`` consistent with the canonical mapping
        self.class_to_idx = self._canonical
        self.classes = [None] * len(self._canonical)
        for cls_name, idx in self._canonical.items():
            if idx >= len(self.classes):
                raise ValueError("Canonical mapping must be zero-indexed and contiguous.")
            self.classes[idx] = cls_name

        if any(cls is None for cls in self.classes):
            raise ValueError("Canonical mapping must cover a contiguous range starting at 0.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        cls_name = self._base_classes[label]
        try:
            remapped = self._canonical[cls_name]
        except KeyError as exc:
            raise KeyError(f"Class '{cls_name}' missing from canonical mapping.") from exc
        return img, remapped

def get_train_transform(img_size: int = 224, policy: str = "light", elastic: bool = False):
    """Create an albumentations transform for training data."""
    
    if policy == "none":
        aug_list = [A.Resize(img_size, img_size)]
    else:
        if policy not in AUG_POLICIES:
            raise KeyError(f"Unknown augmentation policy '{policy}'. Available options: {list(AUG_POLICIES.keys())} or 'none'.")

        probs = AUG_POLICIES.get(policy)

        aug_list = [
            A.Resize(img_size, img_size),
            A.Rotate(limit=15, p=probs["rotate"]),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=0, p=probs["shift_scale"]),
            A.RandomBrightnessContrast(p=probs["brightness_contrast"]),
            A.CLAHE(p=probs["clahe"]),
            A.RandomGamma(p=probs["gamma"]),
            A.GaussNoise(p=probs["noise"]),
            A.MotionBlur(p=probs["motion_blur"]),
            A.MedianBlur(blur_limit=3, p=probs["median_blur"]),
            A.CoarseDropout(p=probs["dropout"]),
        ]

        if elastic:
            aug_list.append(A.ElasticTransform(p=0.1))

    aug_list += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug_list)


def get_val_test_transform(img_size: int = 224):
    """Validation and test transforms - resize and normalize."""

    return A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),ToTensorV2()])


__all__ = [
    "get_train_transform",
    "get_val_test_transform",
    "infer_patient_id",
    "split_by_patient",
    "make_loaders",
    "compute_class_counts",
    "make_sample_weights_from_counts",
    "RemapTargetsDataset",
]

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

def make_loaders(root_dir: str, batch_size: int = 32, num_workers: int = 2, img_size: int = 224, seed: int = 42, aug: str = "light", pin_memory: bool = True):
    if aug != "none" and aug not in AUG_POLICIES:
        raise ValueError(f"Unknown augmentation policy '{aug}'. Available options: {list(AUG_POLICIES.keys())} or 'none'.")
    
    root = Path(root_dir)
    for s in ("train", "val", "test"):
        if not (root / s).exists():
            raise FileNotFoundError(f"Expected '{s}' folder under {root}. Found only {list(p.name for p in root.iterdir())}")
        
    ds_train = datasets.ImageFolder(root / "train")
    ds_val   = datasets.ImageFolder(root / "val")
    ds_test  = datasets.ImageFolder(root / "test")
    
    canonical = dict(ds_train.class_to_idx)
    if ds_val.class_to_idx != canonical:
        ds_val = RemapTargetsDataset(ds_val, canonical)
    if ds_test.class_to_idx != canonical:
        ds_test = RemapTargetsDataset(ds_test, canonical)

    # 3) Wrap with your Albumentations pipelines
    train_tf = AlbumentationsTransform(get_train_transform(img_size=img_size, policy=aug))
    eval_tf  = AlbumentationsTransform(get_val_test_transform(img_size=img_size))

    wrapped = { "train": TransformDataset(ds_train, train_tf), "val":   TransformDataset(ds_val,   eval_tf), "test":  TransformDataset(ds_test,  eval_tf)}

    loaders = { k: DataLoader(wrapped[k], batch_size=batch_size, shuffle=(k == "train"), num_workers=num_workers, pin_memory=pin_memory)
        for k in wrapped
    }
    return loaders, canonical

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
    
