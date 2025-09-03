
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from .model import create_model
from .utils import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint

def _focus_heatmap(cam: np.ndarray, sigma: int = 12) -> np.ndarray:
    """Sharpen heatmap around its maximum activation.

    Grad-CAM can produce very diffuse maps which are hard to interpret for
    X-ray images.  To emphasise the most discriminative region we centre a
    Gaussian filter on the maximum activation and renormalise the heatmap.  A
    smaller ``sigma`` results in a more localised peak.

    Args:
        cam: 2D heatmap array in the range [0, 1].
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        A new heatmap with activations focused around the peak location.
    """

    h, w = cam.shape
    # Location of the highest response
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    yy, xx = np.mgrid[:h, :w]
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    gaussian /= gaussian.max()
    cam = cam * gaussian
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam

def preprocess(img_path: str, img_size: int = 224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("L").convert("RGB")
    return tfm(img).unsqueeze(0), np.array(img)

def gradcam_on_image(model, img_tensor, target_layer, focus_sigma: int = 12):
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

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
    cam = F.relu(cam)[0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = _focus_heatmap(cam, sigma=focus_sigma)
    return cam, pred_class

def overlay_heatmap(orig_img_bgr, cam, alpha=0.35):
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
    overlay = (alpha * heatmap + (1 - alpha) * orig_img_bgr).astype(np.uint8)

    # overlay = (alpha*heatmap + (1-alpha)*orig_img_bgr).astype(np.uint8)
    return overlay

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out_path", type=str, default="gradcam_overlay.png")
    parser.add_argument("--focus_sigma", type=int, default=12, help="Standard deviation of Gaussian for focusing the heatmap")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    arch = ckpt.get("arch","resnet18")
    model = create_model(num_classes=len(ckpt.get("class_to_idx", {0:'Normal',1:'Pneumonia'})), arch=arch, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x, orig = preprocess(args.image_path, img_size=ckpt.get("img_size", args.img_size))
    # pick a last conv/transformer layer based on arch
    target_layer = None
    if arch == "resnet18":
        target_layer = model.backbone.layer4[-1]
    elif arch == "resnet50":
        target_layer = model.backbone.layer4[-1]
    elif arch == "densenet121":
        target_layer = model.backbone.features.norm5
    elif arch == "mobilenet_v2":
        target_layer = model.backbone.features[-1]
    elif arch == "mobilenet_v3":
        target_layer = model.backbone.features[-1]
    elif arch == "efficientnet_b0":
        target_layer = model.backbone.features[-1]
    elif arch == "vit_b_16":
        raise NotImplementedError("Grad-CAM not implemented for vit_b_16")
    else:
        raise ValueError(f"Unknown arch for Grad-CAM: {arch}")

    cam, pred_class = gradcam_on_image(model, x, target_layer=target_layer, focus_sigma=args.focus_sigma)
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(orig_bgr, cam)
    cv2.imwrite(args.out_path, overlay)
    print(f"Saved Grad-CAM to {args.out_path}. Pred class id: {pred_class}")

if __name__ == "__main__":
    main()
