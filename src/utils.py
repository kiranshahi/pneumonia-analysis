
import random
from typing import Optional
import numpy as np
import torch
from torch import nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint handling PyTorch's weights_only default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # For PyTorch versions <2.6 that don't support weights_only
        return torch.load(path, map_location=map_location)

def class_weights_from_counts(class_counts, normalize: bool = True):
    """Compute class weights from counts.

    Weights are calculated as total / (num_classes * counts). Optionally
    normalize so that the weights sum to the number of classes.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float)
    num_classes = counts.numel()
    total = counts.sum()
    weights = total / (num_classes * counts)
    if normalize:
        weights = weights * num_classes / weights.sum()
    return weights

def focal_alpha_from_counts(class_counts, mode: str = "inv_freq"):
    """Derive alpha values for focal loss from class counts.

    Inverse-frequency alphas are normalized to sum to ``num_classes``.
    """
    if mode == "inv_freq":
        return class_weights_from_counts(class_counts, normalize=True)
    raise ValueError(f"Unsupported mode: {mode}")

class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target = target.long()
        logpt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss