
import random
import numpy as np
import torch

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