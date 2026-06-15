"""
core/i18n.py
============
Internationalisation (i18n) for the Checkered Fabric AOI Digital Twin
dashboard. Supported languages: English (en) - Italian (it)

Usage in any page
-----------------
    from core.i18n import t, lang_selector, get_lang
    lang_selector()          # renders the lang radio in the sidebar
    st.title(t("eda_title")) # looks up the current language automatically
"""

from __future__ import annotations
import streamlit as st

# ── Translation dictionary ─────────────────────────────────────────────────────

_T: dict[str, dict[str, str]] = {

# ══════════════════════════════════════════════════════════════════════════════
# SHARED
# ══════════════════════════════════════════════════════════════════════════════
"lang_label": {"en": "🌐 Language", "it": "🌐 Lingua"},
"err_model_not_found": {
    "en": "Model weights not found at {path}. Run `python scripts/train.py` first.",
    "it": "Pesi del modello non trovati in {path}. Esegui prima `python scripts/train.py`.",
},

# ══════════════════════════════════════════════════════════════════════════════
# APP.PY — Landing page
# ══════════════════════════════════════════════════════════════════════════════
"app_title": {"en": "🧵 Checkered Fabric AOI Digital Twin", "it": "🧵 Gemello Digitale AOI Tessuto a Quadretti"},
"app_caption": {"en": "Smart Factories — Project Work — Tor Vergata", "it": "Smart Factories — Progetto del Corso — Tor Vergata"},

"app_met_dataset": {"en": "Dataset", "it": "Dataset"},
"app_met_dataset_val": {"en": "150 images", "it": "150 immagini"},
"app_met_dataset_delta": {"en": "50 per class", "it": "50 per classe"},
"app_met_model": {"en": "Model", "it": "Modello"},
"app_met_model_val": {"en": "ResNet18", "it": "ResNet18"},
"app_met_model_delta": {"en": "frozen backbone, 3-class head", "it": "backbone congelata, testa a 3 classi"},
"app_met_cv": {"en": "Avg. 5-fold CV accuracy", "it": "Accuratezza media CV a 5-fold"},
"app_met_cv_val": {"en": "~94%", "it": "~94%"},
"app_met_cv_delta": {"en": "see Model Performance", "it": "vedi Prestazioni del Modello"},

"app_info": {
    "en": "Navigate using the sidebar. If `models/fabric_classifier.pth` is missing, "
          "run `python scripts/train.py` first to train the model and generate the "
          "cross-validation results.",
    "it": "Naviga usando la barra laterale. Se `models/fabric_classifier.pth` non è "
          "presente, esegui prima `python scripts/train.py` per addestrare il modello "
          "e generare i risultati della cross-validation.",
},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
"eda_title": {"en": "📊 Exploratory Data Analysis", "it": "📊 Analisi Esplorativa dei Dati"},
"eda_caption": {"en": "Dataset quality, distributions, and sample inspection", "it": "Qualità del dataset, distribuzioni e ispezione dei campioni"},

"eda_sub1": {"en": "1. Class distribution", "it": "1. Distribuzione delle classi"},
"eda_success": {
    "en": "Total images: **{n}** — perfectly balanced (50/50/50), no class-imbalance correction needed.",
    "it": "Immagini totali: **{n}** — perfettamente bilanciato (50/50/50), nessuna correzione per squilibrio di classe necessaria.",
},
"eda_bar_class": {"en": "Class", "it": "Classe"},
"eda_bar_count": {"en": "Number of images", "it": "Numero di immagini"},
"eda_bar_title": {"en": "Images per class", "it": "Immagini per classe"},

"eda_sub2": {"en": "2. Data quality checks", "it": "2. Controlli di qualità dei dati"},
"eda_met_sizes": {"en": "Unique image sizes", "it": "Dimensioni immagine univoche"},
"eda_met_missing": {"en": "Missing / unreadable files", "it": "File mancanti / illeggibili"},
"eda_met_dupes": {"en": "Duplicate filenames", "it": "Nomi file duplicati"},
"eda_caption_resize": {
    "en": "All images are resized to 224x224 during pre-processing regardless of native size.",
    "it": "Tutte le immagini vengono ridimensionate a 224x224 durante il pre-processing, indipendentemente dalla dimensione originale.",
},

"eda_sub3": {"en": "3. Pixel intensity distributions by class", "it": "3. Distribuzioni di intensità dei pixel per classe"},
"eda_violin_title": {"en": "Mean pixel brightness per image, by class", "it": "Luminosità media dei pixel per immagine, per classe"},
"eda_scatter_title": {"en": "Mean Red vs Mean Blue channel value per image", "it": "Valore medio canale Rosso vs Blu per immagine"},

"eda_sub4": {"en": "4. Sample images", "it": "4. Immagini di esempio"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Defect Classifier
# ══════════════════════════════════════════════════════════════════════════════
"clf_title": {"en": "🔍 Defect Classifier — Digital Twin Inference", "it": "🔍 Classificatore Difetti — Inferenza del Gemello Digitale"},
"clf_caption": {
    "en": "Observable Manufacturing Element → Device Communication Entity → Digital Twin Entity (this page)",
    "it": "Observable Manufacturing Element → Device Communication Entity → Digital Twin Entity (questa pagina)",
},

"clf_sub1": {"en": "1. Choose an image", "it": "1. Scegli un'immagine"},
"clf_radio_source": {"en": "Image source", "it": "Origine immagine"},
"clf_radio_opt1": {"en": "Pick from dataset", "it": "Scegli dal dataset"},
"clf_radio_opt2": {"en": "Upload an image", "it": "Carica un'immagine"},
"clf_select_class": {
    "en": "Class (for browsing — the model does not see this label)",
    "it": "Classe (solo per navigazione — il modello non vede questa etichetta)",
},
"clf_select_file": {"en": "Image file", "it": "File immagine"},
"clf_uploader": {"en": "Upload a fabric image", "it": "Carica un'immagine di tessuto"},
"clf_info_select": {
    "en": "Select or upload an image to run the digital twin inference.",
    "it": "Seleziona o carica un'immagine per eseguire l'inferenza del gemello digitale.",
},

"clf_sub2": {"en": "2. Digital Twin prediction", "it": "2. Predizione del Gemello Digitale"},
"clf_img_input": {"en": "Input: {label}", "it": "Input: {label}"},
"clf_img_resized": {"en": "Resized (224x224, model input)", "it": "Ridimensionata (224x224, input del modello)"},
"clf_img_gradcam": {"en": "Grad-CAM — class: {cls}", "it": "Grad-CAM — classe: {cls}"},
"clf_bar_title": {"en": "Class probabilities", "it": "Probabilità per classe"},
"clf_pred_ok": {"en": "**Prediction: {cls}**\n\nConfidence: {conf}", "it": "**Predizione: {cls}**\n\nConfidenza: {conf}"},
"clf_pred_defect": {"en": "**Prediction: {cls} defect**\n\nConfidence: {conf}", "it": "**Predizione: difetto {cls}**\n\nConfidenza: {conf}"},
"clf_warning_lowconf": {
    "en": "Low confidence (<70%). Per the 'No Defect' SOP, this frame should be flagged for human review rather than auto-classified.",
    "it": "Confidenza bassa (<70%). Secondo la SOP 'No Defect', questo fotogramma dovrebbe essere segnalato per la revisione umana invece di essere classificato automaticamente.",
},
"clf_caption_gradcam": {
    "en": "Grad-CAM highlights the image regions that most influenced the prediction "
          "— use it to sanity-check that the model is reacting to the defect itself "
          "(hole / scratch) and not to lighting or staging artefacts.",
    "it": "Grad-CAM evidenzia le regioni dell'immagine che hanno maggiormente influenzato "
          "la predizione — usalo per verificare che il modello reagisca al difetto stesso "
          "(foro / graffio) e non ad artefatti di illuminazione o di messa in scena.",
},

"clf_sub3": {"en": "3. Quality Assistant — what should I do?", "it": "3. Assistente Qualità — cosa devo fare?"},
"clf_query_template": {
    "en": "What causes a {cls} defect and what is the corrective action?",
    "it": "Quali sono le cause di un difetto \"{cls}\" e qual è l'azione correttiva?",
},
"clf_predicted_sop": {"en": "Predicted **{cls}** defect → relevant SOP:", "it": "Difetto **{cls}** predetto → SOP rilevante:"},
"clf_sources": {"en": "Sources", "it": "Fonti"},
"clf_source_item": {"en": "- **{title}** (relevance {score})", "it": "- **{title}** (rilevanza {score})"},
"clf_open_qa": {"en": "Open full Quality Assistant →", "it": "Apri l'Assistente Qualità completo →"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
"perf_title": {"en": "📈 Model Performance", "it": "📈 Prestazioni del Modello"},
"perf_caption": {
    "en": "Baseline KPIs and validation for the Digital Twin Entity (ResNet18 classifier)",
    "it": "KPI di base e validazione per il Digital Twin Entity (classificatore ResNet18)",
},

"perf_sub1": {"en": "1. 5-fold cross-validation accuracy", "it": "1. Accuratezza cross-validation a 5-fold"},
"perf_met_avg": {"en": "Average accuracy", "it": "Accuratezza media"},
"perf_met_std": {"en": "Std. deviation", "it": "Deviazione standard"},
"perf_met_min": {"en": "Min fold accuracy", "it": "Accuratezza minima fold"},
"perf_met_max": {"en": "Max fold accuracy", "it": "Accuratezza massima fold"},
"perf_bar_title": {"en": "Per-fold test accuracy (30 images/fold)", "it": "Accuratezza di test per fold (30 immagini/fold)"},
"perf_hline": {"en": "average = {val}", "it": "media = {val}"},
"perf_warning": {
    "en": "**Read the spread, not just the average.** With only 30 test images per "
          "fold, a single misclassification shifts accuracy by ~3.3 points. The "
          "{minv}–{maxv} range is expected sampling noise for this dataset size, not "
          "model instability — but it means the 94% headline number should always be "
          "reported with its standard deviation, and more labelled data would tighten "
          "this estimate.",
    "it": "**Considera la dispersione, non solo la media.** Con solo 30 immagini di "
          "test per fold, una singola classificazione errata sposta l'accuratezza di "
          "~3.3 punti. L'intervallo {minv}–{maxv} è il rumore di campionamento atteso "
          "per questa dimensione del dataset, non instabilità del modello — ma significa "
          "che il valore del 94% dovrebbe sempre essere riportato con la sua deviazione "
          "standard, e più dati etichettati restringerebbero questa stima.",
},

"perf_sub2": {"en": "2. Confusion matrix (aggregated across all 5 folds)", "it": "2. Matrice di confusione (aggregata su tutti i 5 fold)"},
"perf_cm_predicted": {"en": "Predicted", "it": "Predetto"},
"perf_cm_true": {"en": "True", "it": "Reale"},
"perf_cm_title": {"en": "Confusion matrix (150 predictions total, 30 per fold x 5 folds)", "it": "Matrice di confusione (150 predizioni totali, 30 per fold x 5 fold)"},
"perf_confusion_caption": {
    "en": "Largest confusion: {n} image(s) with true label **{true}** predicted as "
          "**{pred}**. This matches the 'No Defect' SOP discussion: subtle holes can "
          "resemble normal weave under uneven lighting.",
    "it": "Confusione maggiore: {n} immagine/i con etichetta reale **{true}** predetta/e "
          "come **{pred}**. Questo è coerente con la discussione della SOP 'No Defect': "
          "piccoli foricome possono somigliare alla trama normale con illuminazione non uniforme.",
},
"perf_no_confusion": {
    "en": "No misclassifications in this run — all confusion is in the per-fold variance above.",
    "it": "Nessuna classificazione errata in questa esecuzione — tutta la variabilità è nella varianza per-fold qui sopra.",
},

"perf_sub3": {"en": "3. Classification report (aggregated)", "it": "3. Report di classificazione (aggregato)"},
"perf_sub4": {"en": "4. Critical discussion — limitations and recommendations", "it": "4. Discussione critica — limitazioni e raccomandazioni"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Quality Assistant
# ══════════════════════════════════════════════════════════════════════════════
"qa_title": {"en": "💬 Quality Assistant", "it": "💬 Assistente Qualità"},
"qa_caption": {
    "en": "RAG over a 12-document SOP / standards knowledge base — User Entity (ISO 23247)",
    "it": "RAG su una knowledge base di 12 documenti SOP / standard — User Entity (ISO 23247)",
},
"qa_claude_success": {
    "en": "Claude API key detected — answers are generated by Claude, grounded in retrieved sources.",
    "it": "Chiave API Claude rilevata — le risposte sono generate da Claude, basate sulle fonti recuperate.",
},
"qa_claude_info": {
    "en": "No Claude API key found — answers are composed locally (extractive) from "
          "the retrieved documents. Set ANTHROPIC_API_KEY (env var or "
          ".streamlit/secrets.toml) to upgrade to LLM-generated answers.",
    "it": "Nessuna chiave API Claude trovata — le risposte sono composte localmente "
          "(estrattive) dai documenti recuperati. Imposta ANTHROPIC_API_KEY (variabile "
          "d'ambiente o .streamlit/secrets.toml) per passare a risposte generate da LLM.",
},
"qa_try_example": {"en": "**Try an example:**", "it": "**Prova un esempio:**"},
"qa_chat_placeholder": {"en": "Ask the Quality Assistant...", "it": "Chiedi all'Assistente Qualità..."},
"qa_sources": {"en": "Sources ({n})", "it": "Fonti ({n})"},
"qa_source_relevance": {"en": "**{title}** — relevance {score}", "it": "**{title}** — rilevanza {score}"},
"qa_kb_expander": {"en": "📚 Knowledge base contents (12 documents)", "it": "📚 Contenuto della knowledge base (12 documenti)"},

"qa_ex1": {"en": "What causes a Circle defect and how do I fix it?", "it": "Quali sono le cause di un difetto Circle e come si risolve?"},
"qa_ex2": {"en": "Why might a Line defect be confused with no defect?", "it": "Perché un difetto Line potrebbe essere confuso con l'assenza di difetti?"},
"qa_ex3": {"en": "What is the OME in this digital twin?", "it": "Cos'è l'OME in questo gemello digitale?"},
"qa_ex4": {"en": "How does the AOI station need to be maintained?", "it": "Come deve essere mantenuta la stazione AOI?"},
"qa_ex5": {"en": "How accurate is the model and what are its limitations?", "it": "Quanto è accurato il modello e quali sono i suoi limiti?"},

"qa_voice_caption": {
    "en": "🎙 Click **Voice** and speak your question, or type it below. Toggle "
          "**Read answers aloud** to have responses read back to you — click "
          "**Stop Speaking** at any time to interrupt.",
    "it": "🎙 Clicca su **Voce** e pronuncia la tua domanda, oppure scrivila qui sotto. "
          "Attiva **Leggi le risposte ad alta voce** per farti leggere le risposte — "
          "clicca su **Ferma lettura** in qualsiasi momento per interromperla.",
},
"qa_tts_toggle": {"en": "🔊 Read answers aloud", "it": "🔊 Leggi le risposte ad alta voce"},
"qa_voice_btn": {"en": "🎙 Voice", "it": "🎙 Voce"},
"qa_stop_speaking": {"en": "⏹ Stop Speaking", "it": "⏹ Ferma lettura"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Live Process
# ══════════════════════════════════════════════════════════════════════════════
"live_title": {"en": "🏭 Live Process — Digital Twin SCADA Floor", "it": "🏭 Processo Live — Pavimento SCADA del Gemello Digitale"},
"live_caption": {
    "en": "Raw Material → Weaving Unit (OME)  →  Digital Twin Hub / QC Scanner (DCE + DT Core)  "
          "→  routing: HRC Packing Cell, Rework Unit, or Shredder/Recycle loop (User Entity)",
    "it": "Materia Prima → Unità di Tessitura (OME)  →  Hub Gemello Digitale / Scanner QC (DCE + DT Core)  "
          "→  instradamento: Cella di Imballaggio HRC, Unità di Rilavorazione o ciclo Trituratore/Riciclo (User Entity)",
},
"live_select_class": {"en": "Class", "it": "Classe"},
"live_select_image": {"en": "Image", "it": "Immagine"},
"live_caption_main": {
    "en": "Selected frame: **{cls}/{fname}** — Digital Twin Core (ResNet18) predicts "
          "**{pred}** at **{conf}** confidence. Click **▶ Run Inspection Cycle** "
          "inside the floor view to animate the part through weaving, the QC scanner, and its "
          "routed destination. Picking a new image reloads the floor for the next cycle. For "
          "**Line** defects, drag the **Rework QA** slider before/during the rework loop: at or "
          "above 70% the repaired part clears the QC Scanner and proceeds to packing/warehouse; "
          "below 70% it is sent back to the Rework Unit and re-checked, looping until the QA "
          "score clears the threshold.",
    "it": "Fotogramma selezionato: **{cls}/{fname}** — il Digital Twin Core (ResNet18) predice "
          "**{pred}** con una confidenza del **{conf}**. Clicca **▶ Run Inspection Cycle** "
          "nella vista del pavimento per animare il pezzo attraverso la tessitura, lo scanner QC e la "
          "destinazione di instradamento. Selezionando una nuova immagine si ricarica il pavimento per "
          "il ciclo successivo. Per i difetti **Line**, sposta il cursore **Rework QA** prima/durante "
          "il ciclo di rilavorazione: al 70% o sopra, il pezzo riparato supera lo scanner QC e procede "
          "verso imballaggio/magazzino; sotto il 70% viene rimandato all'Unità di Rilavorazione e "
          "ricontrollato, in un ciclo che continua finché il punteggio QA non supera la soglia.",
},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Project Reports
# ══════════════════════════════════════════════════════════════════════════════
"rep_title": {"en": "📄 Project Reports & Slides", "it": "📄 Report e Presentazione del Progetto"},
"rep_caption": {
    "en": "Final written report and presentation deck for the Smart Factories course project "
          "work — view inline or download below.",
    "it": "Report scritto finale e presentazione per il progetto del corso Smart Factories "
          "— visualizza qui sotto o scarica.",
},
"rep_tab_report": {"en": "📄 Final Report", "it": "📄 Report Finale"},
"rep_tab_slides": {"en": "📊 Presentation Slides", "it": "📊 Diapositive di Presentazione"},
"rep_download": {"en": "⬇️ Download {name}", "it": "⬇️ Scarica {name}"},
"rep_not_found": {"en": "`{name}` not found in the repository.", "it": "`{name}` non trovato nel repository."},

}


# ── Helper functions ─────────────────────────────────────────────────────────

def get_lang() -> str:
    return st.session_state.get("lang", "en")


def t(key: str, **kwargs) -> str:
    lang = get_lang()
    entry = _T.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def lang_selector() -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    with st.sidebar:
        st.radio(
            t("lang_label"),
            options=["en", "it"],
            format_func=lambda x: "🇬🇧 English" if x == "en" else "🇮🇹 Italiano",
            key="lang",
            horizontal=True,
        )
        st.divider()
