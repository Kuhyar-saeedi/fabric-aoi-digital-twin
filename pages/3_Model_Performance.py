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

from core.i18n import get_lang, lang_selector, t

st.set_page_config(page_title="Model Performance — Checkered Fabric AOI", page_icon="📈", layout="wide")

lang_selector()

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "models" / "cv_results.json"

st.title(t("perf_title"))
st.caption(t("perf_caption"))

if not RESULTS_PATH.exists():
    st.error(t("err_model_not_found", path=RESULTS_PATH))
    st.stop()

with open(RESULTS_PATH) as f:
    res = json.load(f)

classes = res["classes"]

# ── Headline metrics ─────────────────────────────────────────────────────────
st.subheader(t("perf_sub1"))

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(t("perf_met_avg"), f"{res['avg_accuracy']:.1%}")
with c2:
    st.metric(t("perf_met_std"), f"{res['std_accuracy']:.1%}")
with c3:
    st.metric(t("perf_met_min"), f"{min(res['fold_accuracies']):.1%}")
with c4:
    st.metric(t("perf_met_max"), f"{max(res['fold_accuracies']):.1%}")

fold_df = pd.DataFrame({
    "fold": [f"Fold {i+1}" for i in range(len(res["fold_accuracies"]))],
    "accuracy": res["fold_accuracies"],
})
fig = px.bar(fold_df, x="fold", y="accuracy", range_y=[0, 1.05],
              title=t("perf_bar_title"),
              color_discrete_sequence=["#4C78A8"])
fig.add_hline(y=res["avg_accuracy"], line_dash="dash", line_color="red",
               annotation_text=t("perf_hline", val=f"{res['avg_accuracy']:.1%}"))
st.plotly_chart(fig, use_container_width=True)

st.warning(
    t("perf_warning",
      minv=f"{min(res['fold_accuracies']):.0%}",
      maxv=f"{max(res['fold_accuracies']):.0%}"),
    icon="⚠️",
)

# ── Confusion matrix ─────────────────────────────────────────────────────────
st.subheader(t("perf_sub2"))

cm = res["confusion_matrix"]
fig_cm = go.Figure(data=go.Heatmap(
    z=cm, x=classes, y=classes, colorscale="Blues",
    text=cm, texttemplate="%{text}", showscale=True,
))
fig_cm.update_layout(
    xaxis_title=t("perf_cm_predicted"), yaxis_title=t("perf_cm_true"),
    yaxis=dict(autorange="reversed"),
    title=t("perf_cm_title"),
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
        t("perf_confusion_caption", n=off_diag[i, j], true=classes[i], pred=classes[j])
    )
else:
    st.caption(t("perf_no_confusion"))

# ── Classification report ───────────────────────────────────────────────────
st.subheader(t("perf_sub3"))
report = res["classification_report"]
report_df = pd.DataFrame(report).T
report_df = report_df.round(3)
st.dataframe(report_df, use_container_width=True)

# ── Critical discussion ─────────────────────────────────────────────────────
st.divider()
st.subheader(t("perf_sub4"))

if get_lang() == "it":
    st.markdown(f"""
**Dimensione del dataset.** {res['n_images']} immagini totali, 50 per classe. La
cross-validation a 5-fold con {res['epochs_per_fold']} epoche/fold fornisce un
confronto *relativo* solido tra i fold, ma la stima di accuratezza *assoluta* ha
un'ampia incertezza (dev. std = {res['std_accuracy']:.1%}). Un sistema in
produzione richiederebbe centinaia o migliaia di immagini per classe, raccolte
dalla telecamera di ispezione reale in condizioni di illuminazione di produzione.

**Difetti simulati vs. reali.** I difetti "Circle" e "Line" in questo dataset
sono stati creati posizionando oggetti fisici (un foro perforato, un righello/lama)
su tessuto privo di difetti. Questo è un approccio ragionevole per un prototipo, ma
i difetti reali di tessitura (rotture del filo, abrasioni) potrebbero avere firme
di bordo/ombra diverse. Grad-CAM (pagina Defect Classifier) dovrebbe essere usato
per confermare che il modello si concentri sulla regione del difetto stesso, e non
su artefatti di messa in scena come le ombre proiettate dall'oggetto posizionato.

**Backbone congelata.** Viene addestrato solo lo strato lineare finale; la backbone
convoluzionale ResNet18 mantiene i suoi pesi ImageNet. Questo è appropriato per un
dataset di 150 immagini (evita l'overfitting su un numero molto maggiore di
parametri) ma limita l'accuratezza raggiungibile — il fine-tuning di strati più
profondi con più dati potrebbe migliorare la discriminazione tra classi
visivamente simili (Circle vs No defect).

**Singolo motivo del tessuto.** Il modello ha visto solo questo motivo a quadretti
blu/giallo. Distribuirlo su una dimensione/colore di quadretto diverso o su un tipo
di tessitura diverso richiederebbe probabilmente dati etichettati aggiuntivi e un
nuovo addestramento.

**Raccomandazione per il deployment.** Usa una soglia di confidenza (es. 0.7,
segnalata nella pagina Defect Classifier) per instradare le predizioni a basso
confidenza verso la revisione umana, esegui il controllo di calibrazione AOI
settimanale (Quality Assistant — AOI Maintenance), e considera questo valore del
94% +/- {res['std_accuracy']:.1%} come una baseline pilota da rimisurare una volta
raccolte immagini di produzione reali.
""")
else:
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
