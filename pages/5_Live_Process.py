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
from core.i18n import get_lang, lang_selector, t  # noqa: E402

st.set_page_config(page_title="Live Process — Checkered Fabric AOI", page_icon="🏭", layout="wide")

lang_selector()

DATASET_DIR = ROOT / "dataset_web" if (ROOT / "dataset_web").exists() else ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "fabric_classifier.pth"
TEMPLATE_PATH = ROOT / "assets" / "factory_floor_template.html"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title(t("live_title"))
st.caption(t("live_caption"))

if not MODEL_PATH.exists():
    st.error(t("err_model_not_found", path=MODEL_PATH))
    st.stop()


@st.cache_resource
def _load():
    return load_model(MODEL_PATH, device=DEVICE)


model = _load()

# ── Controls ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    cls = st.selectbox(t("live_select_class"), CLASSES)
with c2:
    files = sorted((DATASET_DIR / cls).glob("*.jpg"))
    fname = st.selectbox(t("live_select_image"), [f.name for f in files])

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
    t("live_caption_main", cls=cls, fname=fname, pred=pred_class, conf=f"{confidence:.1%}")
)
