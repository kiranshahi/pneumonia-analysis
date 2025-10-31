import argparse
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from .model import create_model
from .utils import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint
from .gradcam_vit import VITGradCam

def preprocess(img_path: str, img_size: int = 224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("L").convert("RGB")
    return tfm(img).unsqueeze(0), np.array(img)

def gradcam_on_image(model, img_tensor, target_layer):

    activations = []
    gradients = []

    # Forward hook to store activation
    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    # backward hook to capture gradient
    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(img_tensor)
    pred_class = logits.argmax(1).item()
    score = logits[0, pred_class]
    model.zero_grad()
    score.backward()

    h1.remove(); h2.remove()

    act = activations[0]          # [1, C, H, W]
    grad = gradients[0]           # [1, C, H, W]
    weights = grad.mean(dim=(2,3), keepdim=True)   # [1, C, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=False)  # [1, H, W]

    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(1), size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]

    cam = cam.cpu().numpy()
    cam_min = cam.min()
    cam_max = cam.max()
    
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32), pred_class


def overlay_heatmap(orig_img_bgr, cam, alpha=0.35):
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * orig_img_bgr).astype(np.uint8)

    return overlay

def disable_inplace_relu(m: nn.Module):
    for child in m.children():
        if isinstance(child, nn.ReLU):
            child.inplace = False
        disable_inplace_relu(child)




def patch_densenet_forward(model):
    def _densenet_forward_no_inplace(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False) # patched to avoid inplace
        out = F.adaptive_avg_pool2d(out, (1, 1)).reshape(out.size(0), -1)
        return self.backbone.classifier(out)
    model.forward = types.MethodType(_densenet_forward_no_inplace, model)

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out_path", type=str, default="gradcam_overlay.png")
    parser.add_argument("--arch", type=str, required=True)
    
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    arch = args.arch
    model = create_model(num_classes=len(ckpt.get("class_to_idx", {0:'Normal',1:'Pneumonia'})), arch=arch, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    disable_inplace_relu(model)
    if arch.startswith("densenet"):
        patch_densenet_forward(model)

    x, orig = preprocess(args.image_path, img_size=ckpt.get("img_size", args.img_size))
    # last conv/transformer layer based on arch for grad cam
    target_layer = None
    if arch == "resnet18":
        target_layer = model.backbone.layer4[-1]
    elif arch == "resnet50":
        target_layer = model.backbone.layer4[-1]
    elif arch == "densenet121":
        target_layer = model.backbone.features.denseblock4
    elif arch == "mobilenet_v2":
        target_layer = model.backbone.features[-1]
    elif arch == "mobilenet_v3":
        target_layer = model.backbone.features[-1]
    elif arch == "efficientnet_b0":
        target_layer = model.backbone.features[-1]
    elif arch == "vit_b_16":
        # target_layer = model.backbone.encoder.layers[-1].ln_1
        target_layer = model.backbone.encoder.ln
    else:
        raise ValueError(f"Unknown arch for Grad-CAM: {arch}")

    if arch == "vit_b_16":
        vit_cam = VITGradCam(model, target_layer, input_size=224, patch_size=16)
        cam, pred_class = vit_cam(x)
    else:
        cam, pred_class = gradcam_on_image(model, x, target_layer=target_layer)
    
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.resize(orig_bgr, (cam.shape[1], cam.shape[0]))

    overlay = overlay_heatmap(orig_bgr, cam)
    cv2.imwrite(args.out_path, overlay)
    print(f"Saved Grad-CAM to {args.out_path}. Pred class id: {pred_class}")

if __name__ == "__main__":
    main()