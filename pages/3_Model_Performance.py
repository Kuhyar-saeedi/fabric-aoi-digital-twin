"""
pages/3_Model_Performance.py
=============================
5-fold cross-validation results, confusion matrix, and critical discussion of
model/dataset limitations.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Performance — Checkered Fabric AOI", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "models" / "cv_results.json"

st.title("📈 Model Performance")
st.caption("Baseline KPIs and validation for the Digital Twin Entity (ResNet18 classifier)")

if not RESULTS_PATH.exists():
    st.error(f"{RESULTS_PATH} not found. Run `python scripts/train.py` first.")
    st.stop()

with open(RESULTS_PATH) as f:
    res = json.load(f)

classes = res["classes"]

# ── Headline metrics ─────────────────────────────────────────────────────────
st.subheader("1. 5-fold cross-validation accuracy")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Average accuracy", f"{res['avg_accuracy']:.1%}")
with c2:
    st.metric("Std. deviation", f"{res['std_accuracy']:.1%}")
with c3:
    st.metric("Min fold accuracy", f"{min(res['fold_accuracies']):.1%}")
with c4:
    st.metric("Max fold accuracy", f"{max(res['fold_accuracies']):.1%}")

fold_df = pd.DataFrame({
    "fold": [f"Fold {i+1}" for i in range(len(res["fold_accuracies"]))],
    "accuracy": res["fold_accuracies"],
})
fig = px.bar(fold_df, x="fold", y="accuracy", range_y=[0, 1.05],
              title="Per-fold test accuracy (30 images/fold)",
              color_discrete_sequence=["#4C78A8"])
fig.add_hline(y=res["avg_accuracy"], line_dash="dash", line_color="red",
               annotation_text=f"average = {res['avg_accuracy']:.1%}")
st.plotly_chart(fig, use_container_width=True)

st.warning(
    "**Read the spread, not just the average.** With only 30 test images per "
    "fold, a single misclassification shifts accuracy by ~3.3 points. The "
    f"{min(res['fold_accuracies']):.0%}–{max(res['fold_accuracies']):.0%} range "
    "is expected sampling noise for this dataset size, not model instability — "
    "but it means the 94% headline number should always be reported with its "
    "standard deviation, and more labelled data would tighten this estimate.",
    icon="⚠️",
)

# ── Confusion matrix ─────────────────────────────────────────────────────────
st.subheader("2. Confusion matrix (aggregated across all 5 folds)")

cm = res["confusion_matrix"]
fig_cm = go.Figure(data=go.Heatmap(
    z=cm, x=classes, y=classes, colorscale="Blues",
    text=cm, texttemplate="%{text}", showscale=True,
))
fig_cm.update_layout(
    xaxis_title="Predicted", yaxis_title="True",
    yaxis=dict(autorange="reversed"),
    title="Confusion matrix (150 predictions total, 30 per fold x 5 folds)",
)
st.plotly_chart(fig_cm, use_container_width=True)

# Identify largest off-diagonal confusion
import numpy as np
cm_arr = np.array(cm)
off_diag = cm_arr.copy()
np.fill_diagonal(off_diag, 0)
if off_diag.sum() > 0:
    i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
    st.caption(
        f"Largest confusion: {off_diag[i, j]} image(s) with true label "
        f"**{classes[i]}** predicted as **{classes[j]}**. This matches the "
        "'No Defect' SOP discussion: subtle holes can resemble normal weave "
        "under uneven lighting."
    )
else:
    st.caption("No misclassifications in this run — all confusion is in the per-fold variance above.")

# ── Classification report ───────────────────────────────────────────────────
st.subheader("3. Classification report (aggregated)")
report = res["classification_report"]
report_df = pd.DataFrame(report).T
report_df = report_df.round(3)
st.dataframe(report_df, use_container_width=True)

# ── Critical discussion ─────────────────────────────────────────────────────
st.divider()
st.subheader("4. Critical discussion — limitations and recommendations")

st.markdown(f"""
**Dataset size.** {res['n_images']} images total, 50 per class. 5-fold CV with
{res['epochs_per_fold']} epochs/fold gives a robust *relative* comparison
across folds, but the *absolute* accuracy estimate has wide uncertainty
(std = {res['std_accuracy']:.1%}). A production system would need hundreds to
thousands of images per class, collected from the real inspection camera
under production lighting.

**Staged vs. real defects.** "Circle" and "Line" defects in this dataset were
created by placing physical objects (a punched hole, a ruler/blade) on good
fabric. This is a reasonable approach for a prototype, but real weaving
defects (yarn pull-outs, abrasion) may have different edge/shadow signatures.
Grad-CAM (Defect Classifier page) should be used to confirm the model attends
to the defect region itself, not to staging artefacts such as shadows cast by
the placed object.

**Frozen backbone.** Only the final linear layer is trained; the ResNet18
convolutional backbone keeps its ImageNet weights. This is appropriate for a
150-image dataset (avoids overfitting a much larger number of parameters) but
caps achievable accuracy — fine-tuning deeper layers with more data could
improve discrimination between visually similar classes (Circle vs No defect).

**Single fabric pattern.** The model has only seen this one blue/yellow
checkered pattern. Deploying it on a different check size/colour or weave
type would likely require additional labelled data and retraining.

**Recommendation for deployment.** Use a confidence threshold (e.g. 0.7,
flagged on the Defect Classifier page) to route low-confidence predictions to
human review, run the Weekly AOI calibration check (Quality Assistant — AOI
Maintenance), and treat this 94% +/- {res['std_accuracy']:.1%} figure as a
pilot baseline to be re-measured once real production images are collected.
""")
