"""
app.py
======
Landing page for the Checkered Fabric AOI Digital Twin.

Smart Factories course project (Tor Vergata) — combines:
  Track 2: Digital Twin, Data-driven (Quality monitoring)
  Track 3: RAG for Industrial Knowledge (Quality knowledge base)
mapped to ISO 23247.
"""

import streamlit as st

st.set_page_config(
    page_title="Checkered Fabric AOI Digital Twin",
    page_icon="🧵",
    layout="wide",
)

st.title("🧵 Checkered Fabric AOI Digital Twin")
st.caption("Smart Factories — Project Work — Tor Vergata")

st.markdown("""
## The case

A **gingham/checkered cotton fabric weaving line** is fitted with an
**Automated Optical Inspection (AOI)** station at the finishing/inspection
frame. A camera images the moving fabric web; each image patch must be
classified as:

- 🔵 **Circle** — a circular hole / puncture in the weave
- 〜 **Line** — a linear scratch / cut mark across the weave
- ✅ **No defect** — a clean, regular checkered pattern

This project builds a **data-driven Digital Twin** of that inspection station
(Track 2 — Quality Monitoring) and a **RAG-based Quality Assistant**
(Track 3 — Industrial Knowledge Management) that explains *why* a defect
occurred and *what to do about it*.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset", "150 images", "50 per class")
with col2:
    st.metric("Model", "ResNet18", "frozen backbone, 3-class head")
with col3:
    st.metric("Avg. 5-fold CV accuracy", "~94%", "see Model Performance")

st.divider()

st.markdown("""
## ISO 23247 mapping

| ISO 23247 Entity | This project |
|---|---|
| **Observable Manufacturing Element (OME)** | The fabric web at the inspection frame |
| **Device Communication Entity (DCE)** | AOI camera (images in `dataset/` stand in for a live feed) |
| **Data Collection & Device Control** | Pre-processing pipeline (resize, normalise, augment) |
| **Digital Twin Entity (Core)** | ResNet18 classifier + Grad-CAM explainability |
| **User Entity** | This dashboard — Defect Classifier + Quality Assistant |
| **Cross System Entity** | MES / QMS for traceability (out of scope — discussed conceptually) |

## How to use this dashboard

1. **EDA** — explore the dataset: class balance, sample images, basic image statistics.
2. **Defect Classifier** — pick or upload a fabric image, get a prediction, confidence
   scores, and a Grad-CAM explainability heatmap.
3. **Model Performance** — 5-fold cross-validation results, confusion matrix, and a
   critical discussion of limitations.
4. **Quality Assistant** — ask natural-language questions about defect causes,
   corrective actions, the ISO 23247 mapping, and quality standards. Answers are
   retrieved (with sources) from a small SOP knowledge base.
""")

st.info(
    "Navigate using the sidebar. If `models/fabric_classifier.pth` is missing, "
    "run `python scripts/train.py` first to train the model and generate the "
    "cross-validation results.",
    icon="ℹ️",
)
