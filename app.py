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

from core.i18n import get_lang, lang_selector, t

st.set_page_config(
    page_title="Checkered Fabric AOI Digital Twin",
    page_icon="🧵",
    layout="wide",
)

lang_selector()

st.title(t("app_title"))
st.caption(t("app_caption"))

if get_lang() == "it":
    st.markdown("""
## Il caso

Una **linea di tessitura di tessuto di cotone a quadretti (gingham)** è
equipaggiata con una stazione di **Ispezione Ottica Automatica (AOI)**
al telaio di finitura/ispezione. Una telecamera riprende il tessuto in
movimento; ogni porzione di immagine deve essere classificata come:

- 🔵 **Circle** — un foro circolare / una perforazione nella trama
- 〜 **Line** — un graffio lineare / un taglio lungo la trama
- ✅ **No defect** — un motivo a quadretti regolare e privo di difetti

Questo progetto costruisce un **Gemello Digitale data-driven** di questa
stazione di ispezione (Track 2 — Monitoraggio della Qualità) e un
**Assistente Qualità basato su RAG** (Track 3 — Gestione della Conoscenza
Industriale) che spiega *perché* si è verificato un difetto e *cosa fare*
in merito.
""")
else:
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
    st.metric(t("app_met_dataset"), t("app_met_dataset_val"), t("app_met_dataset_delta"))
with col2:
    st.metric(t("app_met_model"), t("app_met_model_val"), t("app_met_model_delta"))
with col3:
    st.metric(t("app_met_cv"), t("app_met_cv_val"), t("app_met_cv_delta"))

st.divider()

if get_lang() == "it":
    st.markdown("""
## Mappatura ISO 23247

| Entità ISO 23247 | In questo progetto |
|---|---|
| **Observable Manufacturing Element (OME)** | Il tessuto al telaio di ispezione |
| **Device Communication Entity (DCE)** | Telecamera AOI (le immagini in `dataset/` simulano un feed live) |
| **Data Collection & Device Control** | Pipeline di pre-processing (ridimensionamento, normalizzazione, augmentation) |
| **Digital Twin Entity (Core)** | Classificatore ResNet18 + spiegabilità Grad-CAM |
| **User Entity** | Questa dashboard — Classificatore Difetti + Assistente Qualità |
| **Cross System Entity** | MES / QMS per la tracciabilità (fuori scope — discusso concettualmente) |

## Come usare questa dashboard

1. **EDA** — esplora il dataset: bilanciamento delle classi, immagini di esempio, statistiche di base.
2. **Defect Classifier** — scegli o carica un'immagine di tessuto, ottieni una predizione,
   i punteggi di confidenza e una mappa di calore Grad-CAM per la spiegabilità.
3. **Model Performance** — risultati della cross-validation a 5-fold, matrice di confusione
   e una discussione critica delle limitazioni.
4. **Quality Assistant** — fai domande in linguaggio naturale sulle cause dei difetti,
   le azioni correttive, la mappatura ISO 23247 e gli standard di qualità. Le risposte sono
   recuperate (con fonti) da una piccola knowledge base di SOP.
5. **Live Process** — un pavimento di fabbrica animato in stile SCADA (OME → DCE → DT Core
   → User Entity). Scegli un fotogramma, esegui un ciclo di ispezione e osservalo essere
   tessuto, scansionato dal Digital Twin Hub (ResNet18) e instradato verso la cella di
   imballaggio HRC, l'unità di rilavorazione o il ciclo trituratore/riciclo in base alla predizione.
6. **Project Reports** — visualizza e scarica il report scritto finale e le diapositive
   della presentazione direttamente dalla dashboard.
""")
else:
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
5. **Live Process** — an animated SCADA-style factory floor (OME → DCE → DT Core
   → User Entity). Pick a frame, run an inspection cycle, and watch it woven,
   scanned by the Digital Twin Hub (ResNet18), and routed to the HRC packing
   cell, the rework unit, or the shredder/recycle loop depending on the
   prediction.
6. **Project Reports** — view and download the final written report and
   presentation slides directly from the dashboard.
""")

st.info(t("app_info"), icon="ℹ️")
