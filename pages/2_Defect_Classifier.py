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
from core.i18n import get_lang, lang_selector, t  # noqa: E402

st.set_page_config(page_title="Defect Classifier — Checkered Fabric AOI", page_icon="🔍", layout="wide")

lang_selector()

DATASET_DIR = ROOT / "dataset_web" if (ROOT / "dataset_web").exists() else ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "fabric_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title(t("clf_title"))
st.caption(t("clf_caption"))

if not MODEL_PATH.exists():
    st.error(t("err_model_not_found", path=MODEL_PATH))
    st.stop()


@st.cache_resource
def _load():
    model = load_model(MODEL_PATH, device=DEVICE)
    cam = GradCAM(model)
    return model, cam


model, gradcam = _load()

# ── Image source ─────────────────────────────────────────────────────────────
st.subheader(t("clf_sub1"))
source = st.radio(t("clf_radio_source"), [t("clf_radio_opt1"), t("clf_radio_opt2")], horizontal=True)

image = None
image_label = None

if source == t("clf_radio_opt1"):
    cls = st.selectbox(t("clf_select_class"), CLASSES)
    files = sorted((DATASET_DIR / cls).glob("*.jpg"))
    fname = st.selectbox(t("clf_select_file"), [f.name for f in files])
    if fname:
        path = DATASET_DIR / cls / fname
        image = Image.open(path)
        image_label = f"{cls}/{fname}"
else:
    uploaded = st.file_uploader(t("clf_uploader"), type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        image_label = uploaded.name

if image is None:
    st.info(t("clf_info_select"))
    st.stop()

# ── Inference ────────────────────────────────────────────────────────────────
st.subheader(t("clf_sub2"))

pred_class, probs = predict(model, image, device=DEVICE)
cam, target_idx = gradcam(image, device=DEVICE)
overlay = overlay_cam(image, cam)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(image.convert("RGB"), caption=t("clf_img_input", label=image_label), use_container_width=True)
with col2:
    st.image(image.convert("RGB").resize((224, 224)), caption=t("clf_img_resized"), use_container_width=True)
with col3:
    st.image(overlay, caption=t("clf_img_gradcam", cls=CLASSES[target_idx]), use_container_width=True)

prob_df = pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
fig = px.bar(prob_df, x="class", y="probability", color="class",
              category_orders={"class": CLASSES}, range_y=[0, 1],
              color_discrete_sequence=px.colors.qualitative.Set2,
              title=t("clf_bar_title"))
fig.update_layout(showlegend=False)

c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(fig, use_container_width=True)
with c2:
    confidence = probs[pred_class]
    if pred_class == "No defect":
        st.success(t("clf_pred_ok", cls=pred_class, conf=f"{confidence:.1%}"))
    else:
        st.error(t("clf_pred_defect", cls=pred_class, conf=f"{confidence:.1%}"))

    if confidence < 0.7:
        st.warning(t("clf_warning_lowconf"), icon="⚠️")

st.caption(t("clf_caption_gradcam"))

# ── Link to Quality Assistant ───────────────────────────────────────────────
if pred_class != "No defect":
    st.subheader(t("clf_sub3"))
    query = t("clf_query_template", cls=pred_class)
    st.markdown(t("clf_predicted_sop", cls=pred_class))

    kb = get_knowledge_base(get_lang())
    results = kb.retrieve(query, top_k=2)
    answer = generate_answer(query, [d["content"] for _, d in results], get_lang())
    if answer is None:
        answer = compose_local_answer(query, results, kb)

    with st.container(border=True):
        st.markdown(answer)
        with st.expander(t("clf_sources")):
            for score, doc in results:
                st.markdown(t("clf_source_item", title=doc["title"], score=f"{score:.2f}"))

    st.page_link("pages/4_Quality_Assistant.py", label=t("clf_open_qa"), icon="💬")
