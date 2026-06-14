"""
scripts/generate_report.py
===========================
Generates report.pdf — the written project report for the Smart Factories
course (Tor Vergata, Prof. Annalisa Santolamazza), combining:
  Track 2 — Digital Twin, Data-driven (Quality Monitoring)
  Track 3 — RAG for Industrial Knowledge Management (Quality Assistant)

Run from the project root (after scripts/train.py and scripts/make_assets.py):
    python scripts/generate_report.py

Requires: reportlab
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT_PDF = ROOT / "report.pdf"

with open(ROOT / "models" / "cv_results.json") as f:
    RES = json.load(f)

GREEN  = colors.HexColor("#1E7145")
DGREEN = colors.HexColor("#0B4A2A")
LGREEN = colors.HexColor("#E3F1E8")
LGRAY  = colors.HexColor("#F5F5F5")
DGRAY  = colors.HexColor("#424242")
MGRAY  = colors.HexColor("#757575")
TEAL   = colors.HexColor("#00695C")
ORANGE = colors.HexColor("#E65100")

BASE = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle("Title", fontSize=26, leading=32, alignment=TA_CENTER,
                              textColor=DGREEN, fontName="Helvetica-Bold", spaceAfter=10)
SUBTITLE_STYLE = ParagraphStyle("Subtitle", fontSize=12, leading=16, alignment=TA_CENTER,
                                 textColor=DGRAY, fontName="Helvetica", spaceAfter=5)
H1 = ParagraphStyle("H1", fontSize=15, leading=20, textColor=DGREEN,
                     fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8)
H2 = ParagraphStyle("H2", fontSize=12, leading=16, textColor=TEAL,
                     fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("Body", fontSize=10, leading=15, alignment=TA_JUSTIFY,
                       fontName="Helvetica", spaceBefore=4, spaceAfter=4)
BULLET = ParagraphStyle("Bullet", fontSize=10, leading=14, fontName="Helvetica",
                         leftIndent=18, bulletIndent=6, spaceBefore=2, spaceAfter=2)
CAPTION = ParagraphStyle("Caption", fontSize=9, leading=12, alignment=TA_CENTER,
                          fontName="Helvetica-Oblique", textColor=MGRAY)
NOTE = ParagraphStyle("Note", fontSize=9, leading=13, alignment=TA_LEFT,
                       fontName="Helvetica-Oblique", textColor=MGRAY,
                       leftIndent=12, spaceBefore=2, spaceAfter=6)
TRACK_LABEL = ParagraphStyle("TrackLabel", fontSize=10, leading=14, textColor=colors.white,
                              fontName="Helvetica-Bold", alignment=TA_CENTER)


def hr(color=GREEN):
    return HRFlowable(width="100%", thickness=1, color=color, spaceAfter=8, spaceBefore=8)


def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def p(text): return Paragraph(text, BODY)
def bullet(text): return Paragraph(f"• {text}", BULLET)
def sp(h=0.3): return Spacer(1, h * cm)
def note(text): return Paragraph(text, NOTE)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(GREEN)
    canvas.rect(2 * cm, A4[1] - 1.8 * cm, A4[0] - 4 * cm, 0.35 * cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DGRAY)
    canvas.drawString(2 * cm, A4[1] - 2.4 * cm,
                       "Checkered Fabric AOI Digital Twin  |  Universita degli Studi di Roma Tor Vergata")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 2.4 * cm, "Project Report")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.2 * cm,
                       "Smart Factories  |  Prof. Annalisa Santolamazza  |  A.Y. 2025-2026")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def meta_table(data, col_widths, header_color=GREEN):
    table_data = []
    for r, row in enumerate(data):
        table_data.append([
            Paragraph(str(cell), ParagraphStyle(
                "th" if r == 0 else "td", fontSize=9, leading=13,
                fontName="Helvetica-Bold" if r == 0 else "Helvetica",
                textColor=colors.white if r == 0 else DGRAY,
            ))
            for cell in row
        ])
    tbl = Table(table_data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LGRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def track_badge(text, color):
    tbl = Table([[Paragraph(text, TRACK_LABEL)]], colWidths=[16 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def fitted_image(name, max_width=15.5 * cm):
    from PIL import Image as PILImage
    path = ASSETS / name
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = h / w
    return Image(str(path), width=max_width, height=max_width * ratio)


# 15-query RAG evaluation (matches scripts/eval_rag.py)
RAG_QUERIES = [
    ("What causes a Circle defect and how do I fix it?", "Correct"),
    ("What should I do if a Line defect is detected on the line?", "Correct"),
    ("What does a 'No defect' prediction mean and what is the false-negative risk?", "Correct"),
    ("What is the Observable Manufacturing Element (OME) in this digital twin?", "Correct"),
    ("Why is this project considered a data-driven Digital Twin rather than a simulation?", "Correct"),
    ("What model architecture is used and how was it trained?", "Correct"),
    ("What is the average accuracy of the classifier across 5-fold cross-validation?", "Correct"),
    ("Why does the per-fold accuracy vary between about 83% and 100%?", "Correct"),
    ("What is the 4-point fabric inspection system and how does it relate to this digital twin's output?", "Correct"),
    ("How should the AOI camera and inspection frame be maintained on a daily basis?", "Correct"),
    ("How would this prototype be integrated with a factory MES/QMS in a real deployment?", "Correct"),
    ("What are the main limitations of the dataset used in this project?", "Correct"),
    ("Is a Circle defect more severe than a Line defect?", "Partial"),
    ("What confidence threshold should trigger a human review of a prediction?", "Partial"),
    ("What is the current market price of cotton yarn?", "No answer"),
]

CORPUS_DOCS = [
    "Process Overview — Checkered Fabric Weaving and Inspection Line",
    "ISO 23247 Mapping — Digital Twin Reference Architecture",
    "What Makes This a Data-Driven Digital Twin",
    "Dataset and Model — 150 Images, ResNet18, Grad-CAM",
    "Model Performance — 5-Fold Cross-Validation Results",
    "SOP — Circle Defect (Hole / Puncture)",
    "SOP — Line Defect (Scratch / Cut Mark)",
    "SOP — “No Defect” Classification and False-Negative Risk",
    "Quality Standard Reference — 4-Point Fabric Inspection System",
    "AOI Station Maintenance and Calibration Guide",
    "Deployment and Integration in a Real Industrial System",
    "Critical Discussion — Dataset and Model Limitations",
]


def build_pdf():
    doc = BaseDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=3.0 * cm, bottomMargin=2.5 * cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=on_page)])

    story = []

    # ── TITLE PAGE ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph("Checkered Fabric AOI", TITLE_STYLE))
    story.append(Paragraph("Digital Twin & Quality Assistant", TITLE_STYLE))
    story.append(sp(0.5))
    story.append(Paragraph("Project Report", SUBTITLE_STYLE))
    story.append(sp(0.3))
    story.append(Paragraph("Smart Factories — A.Y. 2025-2026", SUBTITLE_STYLE))
    story.append(Paragraph("Prof. Annalisa Santolamazza — Department of Enterprise Engineering", SUBTITLE_STYLE))
    story.append(sp(0.5))
    story.append(Paragraph("Kuhyar Saeedi · Danial Mahmoody · Davood Jokar", SUBTITLE_STYLE))
    story.append(Paragraph("Mahyar Emami · Nima Shahrokhi", SUBTITLE_STYLE))
    story.append(sp(0.3))
    story.append(Paragraph("Universita degli Studi di Roma Tor Vergata", SUBTITLE_STYLE))
    story.append(sp(0.7))
    story.append(hr())
    story.append(sp(0.4))
    story.append(track_badge(
        "Track 2 — Digital Twin, Data-Driven (Quality Monitoring)", GREEN))
    story.append(sp(0.15))
    story.append(track_badge(
        "Track 3 — RAG for Industrial Knowledge Management (Quality Assistant)", TEAL))
    story.append(PageBreak())

    # ── 1. INTRODUCTION ──────────────────────────────────────────────────────
    story.append(h1("1. Introduction and Case Description"))
    story.append(hr())
    story.append(p(
        "This project combines two of the four available tracks of the Smart Factories "
        "course into a single, coherent industrial case: an <b>Automated Optical Inspection "
        "(AOI) station</b> on a woven, checkered (“gingham”) cotton fabric line."
    ))
    story.append(p(
        "Yarn is woven on a loom into the characteristic blue/yellow checked pattern, "
        "and the fabric web then passes over a finishing/inspection frame before being "
        "rolled onto the final fabric roll. A camera at the inspection frame captures "
        "images of the moving fabric surface, and each captured patch must be classified "
        "into one of three categories:"
    ))
    for b in [
        "<b>Circle</b> — a roughly circular hole / puncture in the weave",
        "<b>Line</b> — a linear scratch, cut, or abrasion mark across the weave",
        "<b>No defect</b> — a clean, regular checkered pattern",
    ]:
        story.append(bullet(b))
    story.append(sp(0.2))
    story.append(p(
        "<b>Track 2 (Digital Twin, Data-driven — Quality Monitoring)</b> is addressed by "
        "training a ResNet18 image classifier on 150 labelled fabric images, evaluating it "
        "with stratified 5-fold cross-validation, and adding Grad-CAM explainability so an "
        "operator can verify what the model is reacting to."
    ))
    story.append(p(
        "<b>Track 3 (RAG for Industrial Knowledge Management — Quality Knowledge Base)</b> "
        "is addressed by a Quality Assistant: a retrieval-augmented system over a 12-document "
        "knowledge base of Standard Operating Procedures (SOPs), the ISO 23247 mapping, model "
        "documentation, and a textile quality standard, so that when the classifier flags a "
        "defect, the operator can immediately ask “what caused this and what should I do?” "
        "and receive a grounded, source-cited answer."
    ))
    story.append(p(
        "The two tracks share the same case and the same dashboard, reflecting how a real "
        "quality-monitoring digital twin and a knowledge-management assistant would be "
        "deployed together at the same workstation."
    ))
    story.append(sp(0.2))
    story.append(p("<b>Deliverable:</b> a 5-page Streamlit dashboard (app.py + pages/1-4), "
                    "runnable locally with <font face='Courier'>streamlit run app.py</font>, "
                    "covering EDA, the defect classifier with Grad-CAM, model performance / "
                    "5-fold CV, and the Quality Assistant."))

    # ── 2. ISO 23247 MAPPING ────────────────────────────────────────────────
    story.append(h1("2. ISO 23247 Reference Architecture Mapping"))
    story.append(hr())
    story.append(p(
        "ISO 23247 (“Automation systems and integration — Digital twin framework for "
        "manufacturing”) defines a reference architecture of five entity types. The table "
        "below maps each entity to this project, and is what makes the prototype a "
        "<i>digital twin</i> in the ISO 23247 sense rather than a standalone ML model: it "
        "separates the physical asset, the sensing layer, the digital representation, and "
        "the user-facing application."
    ))
    iso_table = [
        ["ISO 23247 Entity", "This Project"],
        ["Observable Manufacturing Element (OME)", "The fabric web at the inspection frame"],
        ["Device Communication Entity (DCE)", "AOI camera / frame grabber (images in dataset/ stand in for a live feed)"],
        ["Data Collection & Device Control", "Pre-processing pipeline: resize to 224x224, normalisation, augmentation"],
        ["Digital Twin Entity (Core)", "ResNet18 classifier + Grad-CAM explainability module"],
        ["User Entity", "Streamlit dashboard — Defect Classifier + Quality Assistant pages"],
        ["Cross System Entity", "MES / QMS for traceability (out of scope for the prototype, discussed conceptually in Section 4.5)"],
    ]
    story.append(meta_table(iso_table, [5.5 * cm, 10 * cm]))
    story.append(PageBreak())

    # ── TRACK 2 HEADER ───────────────────────────────────────────────────────
    story.append(track_badge("TRACK 2 — DIGITAL TWIN, DATA-DRIVEN (QUALITY MONITORING)", GREEN))
    story.append(sp(0.3))
    story.append(h1("3. Dataset and Exploratory Data Analysis"))
    story.append(hr())
    story.append(p(
        "The dataset consists of <b>150 images</b> of checkered cotton fabric, perfectly "
        "balanced across the three classes (50 images each). All images are "
        "<b>3072x3072 px</b> photographs of the same blue/yellow gingham fabric sample. "
        "“Circle” and “Line” defects were created by placing a physical object "
        "(a punched hole, a metal ruler/blade) on otherwise good fabric to simulate the "
        "visual signature of those defect types — a practical approach for a small "
        "student prototype, discussed further as a limitation in Section 6."
    ))
    story.append(fitted_image("eda_overview.png"))
    story.append(Paragraph(
        "Figure 1. Left: class distribution (perfectly balanced, 50/50/50). Right: mean "
        "pixel brightness per class — no class shows a systematically different "
        "brightness, so the model cannot rely on a brightness shortcut to distinguish "
        "classes.", CAPTION))
    story.append(sp(0.2))
    story.append(p(
        "<b>Data quality checks:</b> all 150 images are valid, readable, uniquely named, "
        "and share the same resolution (3072x3072 px) and RGB colour mode — no missing "
        "values or corrupt files. All images are resized to 224x224 during pre-processing "
        "regardless of native size."
    ))

    story.append(h1("4. Model and Methodology"))
    story.append(hr())
    story.append(h2("4.1 Architecture"))
    story.append(p(
        "The classifier is a <b>ResNet18</b> pretrained on ImageNet, with the convolutional "
        "backbone <b>frozen</b> and the final fully-connected layer replaced by a fresh "
        "3-class linear head. Only this head is trained, for 8 epochs, with Adam "
        "(lr=1e-3) and cross-entropy loss."
    ))
    story.append(p(
        "<b>Why this choice is appropriate:</b> with only 150 images, training a full "
        "CNN from scratch (or fine-tuning the full ResNet18) would overfit badly. A frozen "
        "ImageNet backbone re-uses generic low/mid-level visual features (edges, textures, "
        "shapes) that transfer well to fabric-defect images, and the model only has to "
        "learn the final 3-class decision boundary — a much smaller and better-posed "
        "learning problem for this dataset size."
    ))
    story.append(h2("4.2 Pre-processing and Augmentation"))
    story.append(p(
        "All images are resized to 224x224 and converted to tensors. During training, "
        "random horizontal flip, random rotation (±10°), and colour jitter "
        "(brightness/contrast) are applied to reduce overfitting on the small dataset; "
        "evaluation uses a deterministic resize-only transform."
    ))
    story.append(h2("4.3 Validation Protocol"))
    story.append(p(
        "The model is evaluated with <b>stratified 5-fold cross-validation</b> "
        "(random_state=42): each fold trains a fresh 3-class head on 120 images (24 per "
        "class x 5) and tests on the remaining 30 (10 per class). This protocol gives a "
        "robust estimate of generalisation performance despite the small dataset, since "
        "every image is used for testing exactly once across the 5 folds."
    ))

    story.append(h1("5. Results"))
    story.append(hr())
    story.append(h2("5.1 Cross-Validation Accuracy"))
    cv_table = [["Fold", "Accuracy"]]
    for i, a in enumerate(RES["fold_accuracies"], 1):
        cv_table.append([f"Fold {i}", f"{a:.1%}"])
    cv_table.append(["Average", f"{RES['avg_accuracy']:.1%}"])
    cv_table.append(["Std. deviation", f"{RES['std_accuracy']:.1%}"])
    t1 = meta_table(cv_table, [2.5 * cm, 2.5 * cm])
    img1 = fitted_image("fold_accuracy.png", max_width=10 * cm)
    row = Table([[t1, img1]], colWidths=[5 * cm, 10.5 * cm])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(row)
    story.append(sp(0.2))
    story.append(p(
        f"The model achieves an <b>average accuracy of {RES['avg_accuracy']:.1%}</b> "
        f"(±{RES['std_accuracy']:.1%}) across the 5 folds, with per-fold accuracy "
        f"ranging from {min(RES['fold_accuracies']):.0%} to {max(RES['fold_accuracies']):.0%}. "
        "With only 30 test images per fold, a single misclassification shifts accuracy by "
        "~3.3 percentage points — the observed spread is expected sampling noise for "
        "this dataset size, not evidence of an unstable model, but it means the headline "
        "94% figure should always be reported together with its standard deviation."
    ))

    story.append(h2("5.2 Confusion Matrix and Classification Report"))
    img2 = fitted_image("confusion_matrix.png", max_width=7 * cm)
    confmat = RES["confusion_matrix"]
    classes = RES["classes"]
    rep = RES["classification_report"]
    rep_table = [["Class", "Prec.", "Recall", "F1", "N"]]
    for c in classes:
        rep_table.append([c, f"{rep[c]['precision']:.2f}", f"{rep[c]['recall']:.2f}",
                           f"{rep[c]['f1-score']:.2f}", f"{int(rep[c]['support'])}"])
    rep_table.append(["macro avg", f"{rep['macro avg']['precision']:.2f}",
                       f"{rep['macro avg']['recall']:.2f}", f"{rep['macro avg']['f1-score']:.2f}",
                       f"{int(rep['macro avg']['support'])}"])
    t2 = meta_table(rep_table, [2.2 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.3 * cm])
    row2 = Table([[img2, t2]], colWidths=[7.5 * cm, 8 * cm])
    row2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(row2)
    story.append(sp(0.2))
    story.append(p(
        f"The largest confusion is {confmat[0][2]} “Circle” images predicted as "
        f"“No defect” and {confmat[2][0]} “No defect” images predicted as "
        f"“Circle” — i.e. confusion is concentrated between these two classes. "
        "“Line” is classified perfectly (precision = recall = 1.00). This matches the "
        "intuition that a small, faint hole can visually resemble a normal weave gap under "
        "uneven lighting, while a linear scratch is a more visually distinctive pattern."
    ))
    story.append(PageBreak())

    story.append(h2("5.3 Explainability — Grad-CAM"))
    story.append(p(
        "Grad-CAM is computed on the last residual block (layer4) of ResNet18, producing a "
        "heatmap that highlights the image regions most responsible for the predicted "
        "class. This gives the operator (User Entity) a way to sanity-check that the model "
        "is reacting to the actual defect — the hole or the scratch — rather than to "
        "lighting or staging artefacts introduced when the defect images were created."
    ))
    story.append(fitted_image("gradcam_panel.png"))
    story.append(Paragraph(
        "Figure 2. Grad-CAM overlays for one sample image per class (jet colormap; warm "
        "colours = higher contribution to the predicted class). The activations are "
        "broad rather than tightly localised on the defect pixels — expected for a "
        "frozen backbone where only the final linear layer is retrained (Section 6) — "
        "and the relatively low confidence on the Circle sample (46%) reflects the "
        "same limitation.", CAPTION))

    story.append(h1("6. Critical Discussion — Track 2"))
    story.append(hr())
    for title, text in [
        ("Dataset size", f"{RES['n_images']} images total (50/class). 5-fold CV with "
         f"{RES['epochs_per_fold']} epochs/fold gives a robust relative comparison across "
         f"folds, but the absolute accuracy estimate has wide uncertainty "
         f"(std = {RES['std_accuracy']:.1%}). A production system would need hundreds to "
         "thousands of images per class, collected from the real inspection camera under "
         "production lighting."),
        ("Staged vs. real defects", "“Circle” and “Line” defects were created by "
         "placing physical objects (a punched hole, a ruler/blade) on good fabric. Real "
         "weaving defects (yarn pull-outs, abrasion) may have different edge/shadow "
         "signatures — e.g. a clean punched hole lacks the frayed edges of a real "
         "yarn-pullout hole. Grad-CAM should be used to confirm the model attends to the "
         "defect region itself, not to shadows cast by the placed object."),
        ("Frozen backbone", "Only the final linear layer is trained; the ResNet18 "
         "convolutional backbone keeps its ImageNet weights. This is appropriate for a "
         "150-image dataset (avoids overfitting a much larger parameter count) but caps "
         "achievable accuracy — fine-tuning deeper layers with more data could improve "
         "discrimination between visually similar classes (Circle vs. No defect)."),
        ("Single fabric pattern", "The model has only seen this one blue/yellow checkered "
         "pattern. Deploying it on a different check size/colour or weave type would "
         "likely require additional labelled data and retraining."),
        ("Integration into a real system", "In production, the AOI camera (DCE) would "
         "stream frames continuously; an edge device would run the same pre-processing + "
         "inference pipeline (core/model.py) at line speed, and each prediction (class, "
         "confidence, Grad-CAM region, roll ID, meterage, timestamp) would be written to "
         "the MES/QMS (Cross System Entity) for traceability and SPC charting. A confidence "
         "threshold (e.g. 0.7-0.8) would route low-confidence predictions to human review "
         "rather than auto-classifying, per the No-Defect SOP discussed in Section 7."),
    ]:
        story.append(KeepTogether([h2(title), p(text)]))

    story.append(PageBreak())

    # ── TRACK 3 HEADER ───────────────────────────────────────────────────────
    story.append(track_badge("TRACK 3 — RAG FOR INDUSTRIAL KNOWLEDGE MANAGEMENT (QUALITY ASSISTANT)", TEAL))
    story.append(sp(0.3))
    story.append(h1("7. Use Case and User Needs"))
    story.append(hr())
    story.append(p(
        "The Quality Assistant addresses the <b>Quality knowledge base</b> objective from "
        "the RAG track: operators and quality engineers need fast, natural-language access "
        "to Standard Operating Procedures (SOPs), corrective actions, and quality standards "
        "without searching through paper manuals."
    ))
    story.append(p("<b>Primary user needs addressed:</b>"))
    for b in [
        "An operator who just received a “Circle” or “Line” prediction from the "
        "Defect Classifier wants to immediately know the likely root causes and the "
        "corrective action — closing the loop from detection to action.",
        "A new operator wants to understand how the digital twin works (ISO 23247 "
        "mapping, model performance, limitations) without reading the full project "
        "documentation.",
        "A quality engineer wants to relate the AOI output to the mill's existing "
        "point-grading quality standard (4-point fabric inspection system).",
        "A technician wants the AOI station's maintenance/calibration checklist on "
        "demand at the inspection frame.",
    ]:
        story.append(bullet(b))

    story.append(h1("8. Corpus Design"))
    story.append(hr())
    story.append(p(
        "The knowledge base consists of <b>12 documents</b>, written for this project as "
        "an illustrative SOP/standards corpus (no real mill manuals were available — see "
        "limitations in Section 10). Documents cover four areas: project/architecture "
        "documentation, defect-specific SOPs, a quality-standard reference, and "
        "maintenance/deployment guidance."
    ))
    corpus_table = [["#", "Document Title"]]
    for i, title in enumerate(CORPUS_DOCS, 1):
        corpus_table.append([str(i), title])
    story.append(meta_table(corpus_table, [1.2 * cm, 14.3 * cm]))
    story.append(sp(0.2))
    story.append(p(
        "<b>Pre-processing:</b> each document is a short (200-400 word) Markdown-formatted "
        "passage with a title. No chunking is needed at this corpus size — each document "
        "is itself a retrievable unit, which keeps citations interpretable (the operator "
        "sees a named SOP, not an anonymous text fragment)."
    ))

    story.append(h1("9. System Architecture"))
    story.append(hr())
    story.append(h2("9.1 Retrieval"))
    story.append(p(
        "Retrieval is implemented with <b>TF-IDF</b> (term frequency × inverse document "
        "frequency, computed from scratch with NumPy — no external retrieval library "
        "required, fully offline). Each query is tokenised, vectorised in the same TF-IDF "
        "space, and ranked against the 12 documents by cosine similarity; the top-3 "
        "documents are returned with their relevance scores. An optional upgrade to "
        "<b>sentence-transformers</b> (all-MiniLM-L6-v2) semantic embeddings is supported "
        "if the package is installed, for better recall on paraphrased queries."
    ))
    story.append(h2("9.2 Generation"))
    story.append(p(
        "By default (no API key), answers are composed <b>locally and extractively</b>: "
        "the system selects the sentences from the top retrieved documents with the "
        "highest token overlap with the query and assembles them under the source "
        "document's title — producing a grounded answer with zero hallucination risk "
        "and no internet dependency. If an <font face='Courier'>ANTHROPIC_API_KEY</font> "
        "is configured, the same retrieved context is instead passed to "
        "<b>Claude (claude-haiku-4-5)</b> with a system prompt instructing it to answer "
        "only from the provided context and to say so explicitly if the context does not "
        "contain the answer — producing a more fluent, synthesised response while "
        "remaining grounded."
    ))
    story.append(h2("9.3 User Interface"))
    story.append(p(
        "The Quality Assistant is a chat interface (Streamlit "
        "<font face='Courier'>st.chat_input</font>) with five example-query buttons, a "
        "persistent chat history, and a “Sources” expander under every answer showing "
        "the retrieved documents and their relevance scores. The Defect Classifier page "
        "also triggers an inline Quality Assistant query automatically whenever a defect "
        "is predicted, with a link to the full chat for follow-up questions."
    ))

    story.append(h1("10. Evaluation — 15 Representative Queries"))
    story.append(hr())
    story.append(p(
        "The Quality Assistant was evaluated on 15 representative queries spanning all "
        "three required categories (the corpus does not contain hallucination-prone "
        "ambiguous content, so no query was labelled “hallucinated” — see discussion "
        "below). Full output is reproducible via "
        "<font face='Courier'>python scripts/eval_rag.py</font>."
    ))
    eval_table = [["#", "Query", "Label"]]
    for i, (q, label) in enumerate(RAG_QUERIES, 1):
        eval_table.append([str(i), q, label])
    story.append(meta_table(eval_table, [0.9 * cm, 12.1 * cm, 2.5 * cm]))
    story.append(sp(0.2))
    summary_table = [["Label", "Count", "Share"],
                      ["Correct", "12", "80%"],
                      ["Partial", "2", "13%"],
                      ["No answer", "1", "7%"],
                      ["Hallucinated", "0", "0%"]]
    story.append(meta_table(summary_table, [4 * cm, 2 * cm, 2 * cm]))
    story.append(sp(0.2))
    story.append(p(
        "<b>Correct (12/15):</b> queries that map cleanly onto a single document in the "
        "corpus — SOPs, ISO 23247 mapping, model performance, maintenance, and "
        "deployment integration — are all retrieved and answered correctly."
    ))
    story.append(p(
        "<b>Partial (2/15):</b> “Is a Circle defect more severe than a Line defect?” "
        "requires combining the severity notes from both defect SOPs, but the "
        "top-ranked document surfaces only one of them. “What confidence threshold "
        "should trigger human review?” touches three documents: the No-Defect SOP, "
        "which gives the only concrete value (0.8); and the Deployment/Integration "
        "and AOI Maintenance docs, which discuss threshold-based routing and alert "
        "tuning without repeating that number. Retrieval surfaces a mix of these, so "
        "the composed answer may state the general routing concept without clearly "
        "anchoring on the single concrete threshold value."
    ))
    story.append(p(
        "<b>No answer (1/15):</b> “What is the current market price of cotton yarn?” "
        "is intentionally out of scope. The system correctly retrieves only low-relevance "
        "documents and does not fabricate a price — the local extractive composer "
        "cannot hallucinate (it only assembles real sentences from the corpus), and the "
        "Claude system prompt explicitly instructs the model to say so if the context "
        "lacks the answer."
    ))
    story.append(p(
        "<b>Hallucinated (0/15):</b> the local extractive answer composer is structurally "
        "incapable of hallucinating — it can only return sentences that exist verbatim "
        "in the corpus. The Claude-backed path is grounded by an explicit "
        "“answer ONLY from the provided context” instruction, which the out-of-scope "
        "query above is designed to test."
    ))
    story.append(PageBreak())

    story.append(h1("11. Benefits, Limitations and Deployment Challenges"))
    story.append(hr())
    story.append(h2("11.1 Benefits"))
    for b in [
        "Closes the loop from detection to action: a defect prediction on the Defect "
        "Classifier page immediately surfaces the relevant SOP, with no manual lookup.",
        "Fully offline by default (TF-IDF + extractive answers) — no API costs, no "
        "internet dependency, no risk of sending proprietary process data to a third party.",
        "Source-cited answers: every response shows which document(s) it came from and "
        "their relevance score, so an operator can verify the answer against the original "
        "SOP text.",
        "Optional upgrade path (Claude API) for more fluent answers without changing the "
        "retrieval layer or the corpus.",
    ]:
        story.append(bullet(b))
    story.append(h2("11.2 Limitations"))
    for b in [
        "The 12-document corpus is <b>illustrative</b>, written for this project rather "
        "than sourced from a real mill's quality manuals, maintenance procedures, or the "
        "full text of textile quality standards (e.g. ASTM D5430).",
        "TF-IDF retrieval is lexical: it can miss paraphrased queries that share no "
        "vocabulary with the target document (mitigated by the optional semantic "
        "embedding upgrade).",
        "Multi-document synthesis is weak — as seen in the two “partial” "
        "evaluation results, information split across documents is not always combined.",
        "No feedback loop: operator corrections to an assistant's answer are not "
        "currently fed back into the corpus or retrieval ranking.",
    ]:
        story.append(bullet(b))
    story.append(h2("11.3 Deployment Challenges"))
    for b in [
        "<b>Corpus maintenance:</b> a real deployment requires a process for keeping SOPs, "
        "standards, and maintenance procedures in the corpus up to date as they are "
        "revised — stale procedures retrieved with high confidence are worse than no "
        "answer.",
        "<b>Access control:</b> some documents (e.g. internal non-conformance reports) "
        "may be commercially sensitive; a real deployment needs role-based access to the "
        "corpus, not a single shared knowledge base.",
        "<b>Integration with the Cross System Entity:</b> in a full deployment, the "
        "assistant would also query live MES/QMS records (e.g. “what were the last 5 "
        "Circle defects on this loom?”), which requires a retrieval layer over "
        "structured data, not just static documents.",
        "<b>On-line availability:</b> the Claude-backed upgrade requires network access "
        "and an API key; the local extractive fallback should remain the default for a "
        "shop-floor HMI with unreliable connectivity.",
    ]:
        story.append(bullet(b))

    # ── 12. CONCLUSIONS ──────────────────────────────────────────────────────
    story.append(h1("12. Conclusions"))
    story.append(hr())
    story.append(p(
        "This project delivered a working Streamlit dashboard implementing a data-driven "
        "Digital Twin (Track 2) and a RAG-based Quality Assistant (Track 3) for a "
        "checkered-fabric AOI station, explicitly mapped to the ISO 23247 reference "
        "architecture."
    ))
    for b in [
        f"ResNet18 (frozen backbone, 3-class head) reaches {RES['avg_accuracy']:.1%} "
        f"(±{RES['std_accuracy']:.1%}) average accuracy on stratified 5-fold "
        "cross-validation over 150 images, with Grad-CAM explainability giving the "
        "operator a per-prediction view of which regions the model attended to.",
        "A 12-document SOP/standards knowledge base, retrieved via TF-IDF and answered "
        "either extractively (offline) or via Claude (optional), achieved 12/15 correct, "
        "2/15 partial, 1/15 no-answer, and 0/15 hallucinated on the required 15-query "
        "evaluation.",
        "The two tracks are integrated in a single 5-page dashboard: a defect prediction "
        "automatically triggers the relevant SOP lookup, demonstrating the "
        "detection-to-action loop that is the practical value of combining quality "
        "monitoring with knowledge management on the factory floor.",
        "Key limitations — small/staged dataset, illustrative SOP corpus, single fabric "
        "pattern — are documented honestly in Sections 6 and 11, together with concrete "
        "recommendations (confidence-threshold human review, retraining pipeline, corpus "
        "sourced from real mill documentation) for moving from prototype to pilot.",
    ]:
        story.append(bullet(b))
    story.append(sp(0.5))
    story.append(hr())
    story.append(Paragraph(
        "Checkered Fabric AOI Digital Twin  |  Universita di Roma Tor Vergata  |  "
        "Smart Factories A.Y. 2025-2026", CAPTION))

    doc.build(story)
    print(f"Report written to: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
