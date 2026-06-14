"""
core/model.py
=============
Checkered-fabric AOI (Automated Optical Inspection) defect classifier.

ResNet18 transfer learning, 3 classes: Circle, Line, No defect.
Provides: dataset transforms, model builder, training (final model + k-fold CV),
inference, and Grad-CAM explainability for the digital twin dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASSES: List[str] = ["Circle", "Line", "No defect"]
IMG_SIZE = 224

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """ResNet18 with frozen backbone and a fresh linear head for num_classes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path: str | Path, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes=len(CLASSES), pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image: Image.Image, device: str = "cpu") -> Tuple[str, Dict[str, float]]:
    """Run inference on a PIL image. Returns (predicted_class, {class: probability})."""
    x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = CLASSES[int(np.argmax(probs))]
    return pred, {c: float(p) for c, p in zip(CLASSES, probs)}


class GradCAM:
    """Grad-CAM on the last residual block (layer4) of ResNet18."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image: Image.Image, device: str = "cpu", target_class: int | None = None):
        """Returns (cam[IMG_SIZE,IMG_SIZE] in [0,1], target_class_index)."""
        x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(device)
        x.requires_grad_(True)

        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1))

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, target_class


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a Grad-CAM heatmap (values in [0,1]) on a PIL image using a jet colormap."""
    import matplotlib.cm as cm

    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    base_arr = np.asarray(base).astype(np.float32) / 255.0

    heatmap = cm.get_cmap("jet")(cam)[:, :, :3]  # drop alpha channel
    blended = (1 - alpha) * base_arr + alpha * heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
