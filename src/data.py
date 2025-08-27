
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
    "light": dict(rotate=0.5, shift_scale=0.5, brightness_contrast=0.5, clahe=0.3, gamma=0.3, noise=0.2, motion_blur=0.1, median_blur=0.1, dropout=0.1),
    "medium": dict(rotate=0.6, shift_scale=0.6, brightness_contrast=0.6, clahe=0.4, gamma=0.4, noise=0.3, motion_blur=0.15, median_blur=0.15, dropout=0.1),
    "strong": dict(rotate=0.7, shift_scale=0.7, brightness_contrast=0.7, clahe=0.5, gamma=0.5, noise=0.4, motion_blur=0.2, median_blur=0.2, dropout=0.15),
}

class AlbumentationsTransform:
    """Convert PIL image → repeated grayscale → Albumentations → tensor."""
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        img = np.array(img.convert("L"))          # grayscale
        img = np.stack([img]*3, axis=-1)          # repeat channels
        return self.aug(image=img)["image"]

class TransformDataset(torch.utils.data.Dataset):
    """Apply a transform to an existing Subset without altering it."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        base = subset.dataset
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

def get_train_transform(img_size: int = 224, policy: str = "light", elastic: bool = False):
    """Create an albumentations transform for training data.
        Parameters
        ----------
        img_size : int
            Target size for ``Resize``.
        policy : str, default "light"
            Key in :data:`AUG_POLICIES` defining augmentation probabilities.
        elastic : bool, default ``False``
            Whether to include a light elastic deformation.
    """
    probs = AUG_POLICIES.get(policy)

    aug_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=probs["rotate"]),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=probs["shift_scale"]
        ),
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

def make_loaders(root_dir: str, batch_size: int = 32, num_workers: int = 2, img_size: int = 224, seed: int = 42, aug: str = "light"):
      """Create ``DataLoader`` objects for train/val/test splits.

    Parameters
    ----------
    root_dir: str
        Root directory containing the dataset structured as for
        ``torchvision.datasets.ImageFolder``.
    aug: str
        Augmentation strength for the training split (``"none"``, ``"light"``,
        or ``"heavy"``).
    """
    # Load base dataset without transform, then split by patient
    base_ds = datasets.ImageFolder(root=root_dir)
    splits = split_by_patient(base_ds, seed=seed)

    # Prepare transforms
    train_tf = AlbumentationsTransform(get_train_transform(img_size=img_size, policy=aug))

    eval_tf = AlbumentationsTransform(get_val_test_transform(img_size=img_size))
    tforms = {"train": train_tf, "val": eval_tf, "test": eval_tf}

    # Wrap subsets with transforms
    wrapped = {k: TransformDataset(v, tforms[k]) for k, v in splits.items()}

    loaders = {
        k: DataLoader(wrapped[k], batch_size=batch_size, shuffle=(k=='train'), num_workers=num_workers, pin_memory=True)
        for k, v in wrapped.items()
    }

    class_to_idx = base_ds.class_to_idx
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