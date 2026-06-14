"""
scripts/train.py
=================
Train the checkered-fabric defect classifier and produce all artefacts the
Streamlit dashboard needs:

  models/fabric_classifier.pth  -- final model (trained on the full 150-image
                                    dataset), used for live predictions/Grad-CAM
  models/cv_results.json        -- 5-fold cross-validation results (per-fold
                                    accuracy, aggregated confusion matrix,
                                    classification report) for the Model
                                    Performance page

Run from the project root:
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import CLASSES, EVAL_TF, TRAIN_TF, build_model  # noqa: E402

DATASET_PATH = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 8
BATCH_SIZE = 16
SEED = 42


def train_one_model(train_loader, epochs=EPOCHS):
    model = build_model(num_classes=len(CLASSES), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f"    epoch {epoch + 1}/{epochs}  loss={running_loss / len(train_loader.dataset):.4f}")

    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def run_cv():
    print(f"Loading dataset from {DATASET_PATH}")
    base = datasets.ImageFolder(str(DATASET_PATH))
    assert base.classes == CLASSES, f"Expected classes {CLASSES}, got {base.classes}"
    labels = [label for _, label in base.samples]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(labels)), labels))

    train_full = datasets.ImageFolder(str(DATASET_PATH), transform=TRAIN_TF)
    test_full = datasets.ImageFolder(str(DATASET_PATH), transform=EVAL_TF)

    fold_accuracies = []
    all_preds_concat, all_labels_concat = [], []

    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {i + 1}/5 ---")
        train_loader = DataLoader(Subset(train_full, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(Subset(test_full, test_idx), batch_size=BATCH_SIZE)

        model = train_one_model(train_loader)
        preds, lbls = evaluate(model, test_loader)

        acc = float(np.mean(preds == lbls))
        fold_accuracies.append(acc)
        all_preds_concat.extend(preds.tolist())
        all_labels_concat.extend(lbls.tolist())
        print(f"    fold {i + 1} accuracy = {acc:.4f}")

    report = classification_report(all_labels_concat, all_preds_concat, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(all_labels_concat, all_preds_concat).tolist()

    results = {
        "classes": CLASSES,
        "fold_accuracies": fold_accuracies,
        "avg_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "classification_report": report,
        "confusion_matrix": cm,
        "n_images": len(base.samples),
        "epochs_per_fold": EPOCHS,
    }

    with open(MODELS_DIR / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n5-fold CV results:")
    print(f"  per-fold accuracy: {[round(a, 4) for a in fold_accuracies]}")
    print(f"  average accuracy:  {results['avg_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
    print(f"  saved to {MODELS_DIR / 'cv_results.json'}")


def train_final_model():
    print(f"\nTraining final model on the full {len(datasets.ImageFolder(str(DATASET_PATH)).samples)}-image dataset...")
    train_full = datasets.ImageFolder(str(DATASET_PATH), transform=TRAIN_TF)
    loader = DataLoader(train_full, batch_size=BATCH_SIZE, shuffle=True)

    model = train_one_model(loader, epochs=EPOCHS)

    out_path = MODELS_DIR / "fabric_classifier.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved final model to {out_path}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    run_cv()
    train_final_model()
    print("\nDone.")
