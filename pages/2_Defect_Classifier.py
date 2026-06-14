"""
pages/2_Defect_Classifier.py
=============================
Digital Twin inference: pick or upload a fabric image, get a prediction,
confidence scores, and a Grad-CAM explainability heatmap. If a defect is
detected, links to the Quality Assistant for the relevant SOP.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import CLASSES, GradCAM, load_model, overlay_cam, predict  # noqa: E402
from core.rag import compose_local_answer, generate_answer, get_knowledge_base  # noqa: E402

st.set_page_config(page_title="Defect Classifier — Checkered Fabric AOI", page_icon="🔍", layout="wide")

DATASET_DIR = ROOT / "dataset_web" if (ROOT / "dataset_web").exists() else ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "fabric_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title("🔍 Defect Classifier — Digital Twin Inference")
st.caption("Observable Manufacturing Element → Device Communication Entity → Digital Twin Entity (this page)")

if not MODEL_PATH.exists():
    st.error(f"Model weights not found at {MODEL_PATH}. Run `python scripts/train.py` first.")
    st.stop()


@st.cache_resource
def _load():
    model = load_model(MODEL_PATH, device=DEVICE)
    cam = GradCAM(model)
    return model, cam


model, gradcam = _load()

# ── Image source ─────────────────────────────────────────────────────────────
st.subheader("1. Choose an image")
source = st.radio("Image source", ["Pick from dataset", "Upload an image"], horizontal=True)

image = None
image_label = None

if source == "Pick from dataset":
    cls = st.selectbox("Class (for browsing — the model does not see this label)", CLASSES)
    files = sorted((DATASET_DIR / cls).glob("*.jpg"))
    fname = st.selectbox("Image file", [f.name for f in files])
    if fname:
        path = DATASET_DIR / cls / fname
        image = Image.open(path)
        image_label = f"{cls}/{fname}"
else:
    uploaded = st.file_uploader("Upload a fabric image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        image_label = uploaded.name

if image is None:
    st.info("Select or upload an image to run the digital twin inference.")
    st.stop()

# ── Inference ────────────────────────────────────────────────────────────────
st.subheader("2. Digital Twin prediction")

pred_class, probs = predict(model, image, device=DEVICE)
cam, target_idx = gradcam(image, device=DEVICE)
overlay = overlay_cam(image, cam)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(image.convert("RGB"), caption=f"Input: {image_label}", use_container_width=True)
with col2:
    st.image(image.convert("RGB").resize((224, 224)), caption="Resized (224x224, model input)", use_container_width=True)
with col3:
    st.image(overlay, caption=f"Grad-CAM — class: {CLASSES[target_idx]}", use_container_width=True)

prob_df = pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
fig = px.bar(prob_df, x="class", y="probability", color="class",
              category_orders={"class": CLASSES}, range_y=[0, 1],
              color_discrete_sequence=px.colors.qualitative.Set2,
              title="Class probabilities")
fig.update_layout(showlegend=False)

c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(fig, use_container_width=True)
with c2:
    confidence = probs[pred_class]
    if pred_class == "No defect":
        st.success(f"**Prediction: {pred_class}**\n\nConfidence: {confidence:.1%}")
    else:
        st.error(f"**Prediction: {pred_class} defect**\n\nConfidence: {confidence:.1%}")

    if confidence < 0.7:
        st.warning(
            "Low confidence (<70%). Per the 'No Defect' SOP, this frame should "
            "be flagged for human review rather than auto-classified.",
            icon="⚠️",
        )

st.caption(
    "Grad-CAM highlights the image regions that most influenced the prediction "
    "— use it to sanity-check that the model is reacting to the defect itself "
    "(hole / scratch) and not to lighting or staging artefacts."
)

# ── Link to Quality Assistant ───────────────────────────────────────────────
if pred_class != "No defect":
    st.subheader("3. Quality Assistant — what should I do?")
    query = f"What causes a {pred_class} defect and what is the corrective action?"
    st.markdown(f"Predicted **{pred_class}** defect → relevant SOP:")

    kb = get_knowledge_base()
    results = kb.retrieve(query, top_k=2)
    answer = generate_answer(query, [d["content"] for _, d in results])
    if answer is None:
        answer = compose_local_answer(query, results, kb)

    with st.container(border=True):
        st.markdown(answer)
        with st.expander("Sources"):
            for score, doc in results:
                st.markdown(f"- **{doc['title']}** (relevance {score:.2f})")

    st.page_link("pages/4_Quality_Assistant.py", label="Open full Quality Assistant →", icon="💬")
