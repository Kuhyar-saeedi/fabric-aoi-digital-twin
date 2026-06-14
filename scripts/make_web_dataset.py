"""
scripts/make_web_dataset.py
============================
Creates a downscaled copy of dataset/ at dataset_web/ for online deployment
(Streamlit Community Cloud, etc.). The full-res 3072x3072 photos (~6MB each,
~850MB total) are far more than the app needs — the model resizes to 224x224
and the EDA/demo pages only display thumbnails.

dataset_web/ is what gets committed to the deployment repo; dataset/ stays
local-only (used for training and report/asset generation).

Run from the project root:
    python scripts/make_web_dataset.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset"
DST = ROOT / "dataset_web"
CLASSES = ["Circle", "Line", "No defect"]

MAX_SIDE = 640
JPEG_QUALITY = 82

total_before = 0
total_after = 0

for cls in CLASSES:
    src_dir = SRC / cls
    dst_dir = DST / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.jpg")):
        total_before += f.stat().st_size
        with Image.open(f) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = MAX_SIDE / max(w, h)
            if scale < 1:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            out_path = dst_dir / f.name
            im.save(out_path, format="JPEG", quality=JPEG_QUALITY)
            total_after += out_path.stat().st_size
    print(f"{cls}: {len(list(dst_dir.glob('*.jpg')))} images written to {dst_dir}")

print(f"\nTotal size: {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")
