
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

def _make_backbone(arch: str, pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        head_attr = "fc"
    elif arch == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        head_attr = "classifier"
    elif arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[-1].in_features
        head_attr = "classifier_seq"
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return backbone, in_features, head_attr

class PneumoniaNet(nn.Module):
    def __init__(self, num_classes: int = 2, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.backbone, in_features, head_attr = _make_backbone(arch, pretrained=pretrained)
        if head_attr == "fc":
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif head_attr == "classifier":
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        else:  # classifier_seq for efficientnet
            layers = list(self.backbone.classifier.children())
            layers[-1] = nn.Linear(in_features, num_classes)
            self.backbone.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes: int = 2, arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    return PneumoniaNet(num_classes=num_classes, arch=arch, pretrained=pretrained)
