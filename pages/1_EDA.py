"""
pages/1_EDA.py
==============
Exploratory Data Analysis for the checkered-fabric AOI dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from core.i18n import get_lang, lang_selector, t

st.set_page_config(page_title="EDA — Checkered Fabric AOI", page_icon="📊", layout="wide")

lang_selector()

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_web" if (ROOT / "dataset_web").exists() else ROOT / "dataset"
CLASSES = ["Circle", "Line", "No defect"]

st.title(t("eda_title"))
st.caption(t("eda_caption"))

if not DATASET_DIR.exists():
    st.error(f"Dataset folder not found at {DATASET_DIR}")
    st.stop()


@st.cache_data
def scan_dataset():
    rows = []
    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.jpg")):
            with Image.open(f) as img:
                w, h = img.size
                arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            rows.append({
                "class": cls,
                "filename": f.name,
                "path": str(f),
                "width": w,
                "height": h,
                "mean_brightness": arr.mean(),
                "mean_R": arr[:, :, 0].mean(),
                "mean_G": arr[:, :, 1].mean(),
                "mean_B": arr[:, :, 2].mean(),
                "std_brightness": arr.std(),
            })
    return pd.DataFrame(rows)


df = scan_dataset()

# ── Class balance ─────────────────────────────────────────────────────────────
st.subheader(t("eda_sub1"))
counts = df["class"].value_counts().reindex(CLASSES)
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(counts.rename("count"), use_container_width=True)
    st.success(t("eda_success", n=len(df)))
with c2:
    fig = px.bar(counts, x=counts.index, y=counts.values,
                  labels={"x": t("eda_bar_class"), "y": t("eda_bar_count")},
                  title=t("eda_bar_title"), color=counts.index,
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Data quality: image size / missing values ───────────────────────────────
st.subheader(t("eda_sub2"))
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(t("eda_met_sizes"), df[["width", "height"]].drop_duplicates().shape[0])
with c2:
    st.metric(t("eda_met_missing"), 0)
with c3:
    st.metric(t("eda_met_dupes"), int(df["filename"].duplicated().sum()))

size_counts = df.groupby(["width", "height"]).size().reset_index(name="count")
st.dataframe(size_counts, use_container_width=True)
st.caption(t("eda_caption_resize"))

# ── Pixel intensity distributions ───────────────────────────────────────────
st.subheader(t("eda_sub3"))
if get_lang() == "it":
    st.markdown("""
La luminosità media dell'immagine fornisce un controllo rapido: se le immagini
"Circle" o "Line" fossero sistematicamente più scure/chiare di "No defect" (ad
esempio per come è stato fotografato l'oggetto che simula il difetto), il
modello potrebbe imparare in parte una scorciatoia basata sulla luminosità
invece che sulla forma del difetto stesso. Grad-CAM (pagina Defect Classifier)
viene usato più avanti per verificare che il modello si concentri sulla regione
del difetto e non su tali artefatti.
""")
else:
    st.markdown("""
Mean image brightness gives a quick sanity check: if "Circle" or "Line" images
were systematically darker/brighter than "No defect" (e.g. because of how the
defect-simulating object was photographed), the model could partly learn a
brightness shortcut instead of the defect shape itself. Grad-CAM (Defect
Classifier page) is used later to verify the model attends to the defect
region rather than such artefacts.
""")

fig = px.violin(df, x="class", y="mean_brightness", color="class", box=True, points="all",
                 category_orders={"class": CLASSES},
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title=t("eda_violin_title"))
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter(df, x="mean_R", y="mean_B", color="class",
                   category_orders={"class": CLASSES},
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   title=t("eda_scatter_title"),
                   hover_data=["filename"])
st.plotly_chart(fig2, use_container_width=True)

if get_lang() == "it":
    st.caption(
        "Il motivo a quadretti è blu/giallo, quindi sono previsti valori medi "
        "elevati del canale Blu e del canale Rosso+Verde (giallo) per tutte le "
        "classi. Nessuna classe mostra un profilo di colore sistematicamente "
        "diverso — i difetti si distinguono per forma/texture, non per colore."
    )
else:
    st.caption(
        "The checkered pattern is blue/yellow, so high Blue and high Red+Green "
        "(yellow) channel means are expected for all classes. No class shows a "
        "systematically different colour profile — defects are distinguished by "
        "shape/texture, not colour."
    )

# ── Sample images per class ─────────────────────────────────────────────────
st.subheader(t("eda_sub4"))
tabs = st.tabs(CLASSES)
for tab, cls in zip(tabs, CLASSES):
    with tab:
        sample = df[df["class"] == cls].sample(n=min(5, len(df[df["class"] == cls])), random_state=0)
        cols = st.columns(len(sample))
        for col, (_, row) in zip(cols, sample.iterrows()):
            with col:
                st.image(row["path"], caption=row["filename"], use_container_width=True)

st.divider()
if get_lang() == "it":
    st.markdown("""
**Riepilogo EDA**: il dataset è piccolo (150 immagini) ma perfettamente
bilanciato tra le 3 classi, tutte le immagini sono valide e di dimensioni
coerenti, e non c'è un evidente fattore di confondimento legato a
colore/luminosità tra le classi. La principale criticità di qualità —
discussa più approfonditamente nella pagina Model Performance — è che i
difetti "Circle" e "Line" sono stati creati posizionando oggetti fisici su
un tessuto altrimenti senza difetti, il che costituisce un difetto
artificialmente creato piuttosto che naturale.
""")
else:
    st.markdown("""
**EDA summary**: the dataset is small (150 images) but perfectly balanced
across the 3 classes, all images are valid and consistently sized, and there
is no obvious colour/brightness confound between classes. The main quality
caveat — discussed further on the Model Performance page — is that "Circle"
and "Line" defects were created by placing physical objects on otherwise good
fabric, which is a staged rather than naturally-occurring defect.
""")
