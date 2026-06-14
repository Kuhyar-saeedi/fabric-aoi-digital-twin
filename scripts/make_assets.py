"""
scripts/make_assets.py
=======================
Generates static image assets (charts, Grad-CAM panels) used by
generate_report.py and generate_slides.py. Run after scripts/train.py.

Run from the project root:
    python scripts/make_assets.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from core.model import CLASSES, GradCAM, load_model, overlay_cam, predict

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

with open(ROOT / "models" / "cv_results.json") as f:
    res = json.load(f)

# ── 1. Fold accuracy bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
folds = [f"Fold {i+1}" for i in range(len(res["fold_accuracies"]))]
bars = ax.bar(folds, res["fold_accuracies"], color="#4C78A8")
ax.axhline(res["avg_accuracy"], color="red", linestyle="--",
           label=f"average = {res['avg_accuracy']:.1%}")
for b, v in zip(bars, res["fold_accuracies"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}",
            ha="center", fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy (30 images/fold)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(ASSETS / "fold_accuracy.png", dpi=150)
plt.close(fig)
print("wrote fold_accuracy.png")

# ── 2. Confusion matrix heatmap ─────────────────────────────────────────────
cm = np.array(res["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=20)
ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (150 predictions, 5 folds)")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(ASSETS / "confusion_matrix.png", dpi=150)
plt.close(fig)
print("wrote confusion_matrix.png")

# ── 3. Grad-CAM panel (one sample per class) ────────────────────────────────
model = load_model(ROOT / "models" / "fabric_classifier.pth")
cam = GradCAM(model)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, cls in zip(axes, CLASSES):
    img_path = sorted((ROOT / "dataset" / cls).glob("*.jpg"))[0]
    img = Image.open(img_path).convert("RGB")
    pred_class, probs = predict(model, img)
    heatmap, target_idx = cam(img)
    overlay = overlay_cam(img, heatmap)
    ax.imshow(overlay)
    ax.set_title(f"True: {cls}\nPred: {pred_class} ({probs[pred_class]:.0%})", fontsize=10)
    ax.axis("off")
fig.suptitle("Grad-CAM Explainability — One Sample per Class", fontsize=12)
fig.tight_layout()
fig.savefig(ASSETS / "gradcam_panel.png", dpi=150)
plt.close(fig)
print("wrote gradcam_panel.png")

# ── 4. EDA: class distribution + brightness ─────────────────────────────────
rows = []
for cls in CLASSES:
    for f in sorted((ROOT / "dataset" / cls).glob("*.jpg")):
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        rows.append((cls, arr.mean()))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

counts = {cls: sum(1 for c, _ in rows if c == cls) for cls in CLASSES}
axes[0].bar(counts.keys(), counts.values(), color=["#4C78A8", "#F58518", "#54A24B"])
axes[0].set_title("Class Distribution (150 images total)")
axes[0].set_ylabel("Number of images")
for i, cls in enumerate(CLASSES):
    axes[0].text(i, counts[cls] + 1, str(counts[cls]), ha="center")

by_class = {cls: [b for c, b in rows if c == cls] for cls in CLASSES}
axes[1].boxplot([by_class[c] for c in CLASSES], labels=CLASSES)
axes[1].set_title("Mean Pixel Brightness by Class")
axes[1].set_ylabel("Mean brightness (0-255)")

fig.tight_layout()
fig.savefig(ASSETS / "eda_overview.png", dpi=150)
plt.close(fig)
print("wrote eda_overview.png")

print("All assets written to", ASSETS)
