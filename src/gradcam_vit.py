# vit_gradcam.py  -- replace existing VITGradCam with this
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class VITGradCam:
    def __init__(self, model, target_module, input_size=224, patch_size=16):
        
        self.model = model.eval()
        self.target = target_module
        self.input_size = input_size
        self.patch_size = patch_size

        self.activations = None   # raw activation tensor captured in forward hook
        self.gradients = None     # raw grad tensor captured in backward hook
        self._handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def _fwd(module, inp, out):
            self.activations = out.detach()

        def _bwd(module, grad_input, grad_output):
            g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            if g is not None:
                self.gradients = g.detach()

        h1 = self.target.register_forward_hook(_fwd)
        try:
            h2 = self.target.register_full_backward_hook(_bwd)
        except Exception:
            # older pytorch fallback
            h2 = self.target.register_backward_hook(lambda m, gi, go: _bwd(m, gi, go))

        self._handlers = [h1, h2]

    def remove_hooks(self):
        for h in self._handlers:
            try:
                h.remove()
            except Exception:
                pass
        self._handlers = []

    def _reshape_to_feature_map(self, tensor):
        """
        Convert token outputs [B, 1+N, C] -> [B, C, H_p, W_p]
        If tensor already 4D [B, C, H, W] return as-is.
        """
        if tensor is None:
            return None

        t = tensor
        if t.dim() == 4:
            # CNN-like: already [B, C, H, W]
            return t
        if t.dim() == 3:
            # ViT tokens: [B, 1 + N_patches, C]
            b, n, c = t.shape
            if n <= 1:
                raise RuntimeError("Unexpected token tensor with <=1 tokens.")
            tokens = t[:, 1:, :]  # remove CLS token -> [B, N, C]
            n_patches = tokens.shape[1]
            # compute H_p, W_p
            hp = int(math.sqrt(n_patches))
            wp = hp
            if hp * wp != n_patches:
                # fallback to input_size // patch_size if sqrt isn't integer
                hp = wp = self.input_size // self.patch_size
            # [B, N, C] -> [B, C, H_p, W_p]
            feats = tokens.permute(0, 2, 1).contiguous().view(b, c, hp, wp)
            return feats
        raise RuntimeError(f"Unsupported tensor dim {t.dim()} for reshape")

    def __call__(self, input_tensor, target_category=None):
        """
        input_tensor: torch.Tensor [B=1, C, H, W]
        target_category: optional int for class index; if None uses predicted class
        Returns: cam_numpy (H_input x W_input) float32 in [0,1], pred_class (int)
        """
        # ensure on same device
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_(True)

        # clear previous captures
        self.activations = None
        self.gradients = None

        # forward
        out = self.model(input_tensor)
        # get prediction index
        if target_category is None:
            if out.dim() == 1:
                pred = int(out.argmax().item())
            else:
                pred = int(out[0].argmax().item())
        else:
            pred = int(target_category)

        # scalar score for backprop
        score = out[0, pred] if out.dim() > 1 else out[pred]

        # backward
        self.model.zero_grad()
        score.backward()

        # get features and grads reshaped to [B, C, H_p, W_p]
        feat = self._reshape_to_feature_map(self.activations)
        grad = self._reshape_to_feature_map(self.gradients)

        if feat is None or grad is None:
            raise RuntimeError("Activations or gradients not captured. Check target module and hooks.")

        # use first example in batch
        feat = feat[0]   # [C, H_p, W_p]
        grad = grad[0]   # [C, H_p, W_p]

        # compute weights: global average pooling over spatial dims
        weights = torch.mean(grad, dim=(1, 2))  # [C]

        # weighted combination of feature maps
        cam_map = torch.sum(feat * weights[:, None, None], dim=0)  # [H_p, W_p]
        cam_map = F.relu(cam_map)

        # upsample to input size
        cam_map = cam_map.unsqueeze(0).unsqueeze(0)  # [1,1,H_p,W_p]
        cam_map = F.interpolate(cam_map, size=(self.input_size, self.input_size),
                                mode="bilinear", align_corners=False)[0, 0]

        cam_numpy = cam_map.detach().cpu().numpy().astype(np.float32)
        # normalize
        mn, mx = cam_numpy.min(), cam_numpy.max()
        if mx > mn:
            cam_numpy = (cam_numpy - mn) / (mx - mn)
        else:
            cam_numpy = np.zeros_like(cam_numpy)

        return cam_numpy, pred

    def __del__(self):
        # ensure hooks cleaned up
        self.remove_hooks()
