"""
pages/5_Live_Process.py
========================
Animated "SCADA" factory-floor view of the digital twin: raw material is
woven, scanned by the Digital Twin Hub (ResNet18 core), and routed —
no-defect parts go straight to the HRC packing cell and warehouse, "Line"
defects loop through the rework unit before packing, and "Circle" defects
are scrapped through the shredder/recycle loop back to raw material.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import CLASSES, load_model, predict  # noqa: E402

st.set_page_config(page_title="Live Process — Checkered Fabric AOI", page_icon="🏭", layout="wide")

DATASET_DIR = ROOT / "dataset_web" if (ROOT / "dataset_web").exists() else ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "fabric_classifier.pth"
TEMPLATE_PATH = ROOT / "assets" / "factory_floor_template.html"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title("🏭 Live Process — Digital Twin SCADA Floor")
st.caption(
    "Raw Material → Weaving Unit (OME)  →  Digital Twin Hub / QC Scanner (DCE + DT Core)  "
    "→  routing: HRC Packing Cell, Rework Unit, or Shredder/Recycle loop (User Entity)"
)

if not MODEL_PATH.exists():
    st.error(f"Model weights not found at {MODEL_PATH}. Run `python scripts/train.py` first.")
    st.stop()


@st.cache_resource
def _load():
    return load_model(MODEL_PATH, device=DEVICE)


model = _load()

# ── Controls ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    cls = st.selectbox("Class", CLASSES)
with c2:
    files = sorted((DATASET_DIR / cls).glob("*.jpg"))
    fname = st.selectbox("Image", [f.name for f in files])

image = Image.open(DATASET_DIR / cls / fname)
pred_class, probs = predict(model, image, device=DEVICE)
confidence = probs[pred_class]

buf = io.BytesIO()
image.convert("RGB").resize((300, 300)).save(buf, format="JPEG", quality=85)
img_b64 = base64.b64encode(buf.getvalue()).decode()

template = TEMPLATE_PATH.read_text(encoding="utf-8")
html = (
    template.replace("__IMG_B64__", img_b64)
    .replace("__PRED_CLASS__", pred_class)
    .replace("__CONF__", f"{confidence:.1%}")
)

components.html(html, height=760, scrolling=False)

st.caption(
    f"Selected frame: **{cls}/{fname}** — Digital Twin Core (ResNet18) predicts "
    f"**{pred_class}** at **{confidence:.1%}** confidence. Click **▶ Run Inspection Cycle** "
    "inside the floor view to animate the part through weaving, the QC scanner, and its "
    "routed destination. Picking a new image reloads the floor for the next cycle. For "
    "**Line** defects, drag the **Rework QA** slider before/during the rework loop: at or "
    "above 70% the repaired part clears the QC Scanner and proceeds to packing/warehouse; "
    "below 70% it is sent back to the Rework Unit and re-checked, looping until the QA "
    "score clears the threshold."
)
