"""
core/rag.py
===========
Retrieval-Augmented Generation knowledge base for the Checkered Fabric AOI
Digital Twin ("Quality Assistant").

Architecture
------------
Retrieval : TF-IDF (always available, fully offline) or sentence-transformers
            (optional, better semantic recall): pip install sentence-transformers
Generation: Local extractive answer (no LLM needed) composed from retrieved
            chunks. If ANTHROPIC_API_KEY is set, Claude is used to generate a
            grounded, source-cited answer from the retrieved context.

Knowledge base
--------------
12 documents covering: the textile process and asset description, the ISO
23247 mapping, the digital twin / model description, model performance and
limitations, defect-specific SOPs (Circle / Line / No defect), a textile
quality standard reference, AOI maintenance, and deployment/integration notes.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np

# ── Optional semantic embeddings ─────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _SMODEL: "_ST | None" = None

    def _get_semantic_model():
        global _SMODEL
        if _SMODEL is None:
            _SMODEL = _ST("all-MiniLM-L6-v2")
        return _SMODEL

    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False

    def _get_semantic_model():
        return None


# ── Knowledge base ────────────────────────────────────────────────────────────

_DOCS: List[dict] = [

{
"id": "process_overview",
"title": "Process Overview — Checkered Fabric Weaving and Inspection Line",
"content": """
The case considered is a woven cotton "gingham"-style checkered fabric line.
Yarn is woven on a loom into the characteristic blue/yellow checked pattern,
then the fabric web passes over a finishing/inspection frame before being
rolled onto the final fabric roll.

An Automated Optical Inspection (AOI) station is positioned at the inspection
frame: a camera captures images of the fabric surface as it moves under
controlled lighting. Each captured frame is analysed by a vision model that
classifies the visible patch into one of three categories:

- "Circle"    — a roughly circular hole / puncture in the weave
- "Line"      — a linear scratch, cut, or abrasion mark across the weave
- "No defect" — a clean, regular checkered pattern

The objective of the digital twin is QUALITY MONITORING: catching fabric
defects in-line (before the roll is finished and shipped), classifying the
defect type so operators can act on the right root cause, and logging defect
rates over time for process improvement (e.g. SPC charts, OEE/quality losses).
"""},

{
"id": "iso23247_mapping",
"title": "ISO 23247 Mapping — Digital Twin Reference Architecture",
"content": """
ISO 23247 (Automation systems and integration — Digital twin framework for
manufacturing) defines a reference architecture with the following entities,
mapped here to the checkered-fabric AOI case:

Observable Manufacturing Element (OME):
  The physical fabric web at the inspection frame — the moving checkered
  fabric surface that is the "thing" being twinned.

Device Communication Entity (DCE):
  The AOI camera and frame grabber. It captures images of the OME and makes
  them available to the digital twin (in this prototype: image files read
  from the dataset/ folder, standing in for a live camera feed).

Data Collection and Device Control Entity:
  The pre-processing pipeline (resize to 224x224, normalisation, augmentation
  during training) that prepares raw images for the Digital Twin Entity, and
  could in a real deployment trigger device actions (e.g. flag a roll segment,
  trigger a reject gate).

Digital Twin Entity (Core):
  The trained ResNet18 classifier plus Grad-CAM explainability module. It
  holds the digital representation of "what a defect looks like" learned from
  150 labelled images, and produces a class + confidence + saliency map for
  each new image.

User Entity:
  The Streamlit dashboard: Defect Classifier page (operator-facing inference +
  explainability) and Quality Assistant page (RAG-based SOP retrieval).

Cross System Entity:
  Out of scope for this prototype, but in a real plant this would be the
  Manufacturing Execution System (MES) / Quality Management System (QMS) that
  the digital twin would report defect events to, and that would hold the
  traceability link between a defect, the fabric roll ID, and the loom that
  produced it.

This mapping is what makes the prototype a "digital twin" in the ISO 23247
sense rather than a standalone ML model: it explicitly separates the physical
asset (OME), the sensing/communication layer (DCE), the digital
representation (Core), and the user-facing application (User Entity).
"""},

{
"id": "digital_twin_concept",
"title": "What Makes This a Data-Driven Digital Twin",
"content": """
This project follows the data-driven Digital Twin track: instead of a
first-principles simulation of the loom, the digital twin is learned directly
from operational data (images of the fabric surface, some defective, some
not).

Why data-driven is appropriate here:
- Defect appearance (a hole, a scratch) is visually defined, not governed by
  a tractable physical equation — a vision model is the natural fit.
- The relationship between "what the camera sees" and "is this a reject" is
  exactly the mapping a CNN classifier learns.
- The model can be retrained as new defect types or fabric patterns are
  introduced, without re-deriving a physical model.

Digital twin loop in this prototype:
1. OME (fabric web) -> DCE (camera) produces an image.
2. Digital Twin Entity (ResNet18) predicts a class and confidence.
3. Grad-CAM produces a saliency map showing WHERE in the image the decision
   was made — this is the "explainability" layer that lets an operator trust
   (or challenge) the twin's output.
4. If a defect is predicted, the User Entity (Quality Assistant) retrieves the
   relevant SOP so the operator immediately knows the likely cause and the
   corrective action — closing the loop from detection to action.

This is the "Quality monitoring" objective named in the data-driven Digital
Twin track: identify process parameters / conditions (here: visual defect
signatures) that influence product quality, using real operational data
(the 150-image dataset).
"""},

{
"id": "dataset_and_model",
"title": "Dataset and Model — 150 Images, ResNet18, Grad-CAM",
"content": """
Dataset: 150 images of checkered fabric, organised into 3 balanced classes
(50 images each): "Circle", "Line", "No defect". Images are photographs of a
real checkered fabric sample; "Circle" and "Line" defects were created by
placing a physical object (a punched hole, a metal ruler/blade) on the fabric
to simulate the visual signature of those defect types.

Pre-processing: all images resized to 224x224 and converted to tensors.
Training augmentation: random horizontal flip, random rotation (+/-10 deg),
and colour jitter (brightness/contrast) to reduce overfitting on a small
dataset.

Model: ResNet18 pretrained on ImageNet, backbone frozen, final fully-connected
layer replaced with a 3-class linear head and fine-tuned for 8 epochs with
Adam (lr=1e-3) and cross-entropy loss. ResNet18 with a frozen backbone is the
standard go-to choice for small datasets / constrained compute: it reuses
generic visual features (edges, textures, shapes) learned from ImageNet and
only learns the final decision boundary from the 150 available images.

Explainability: Grad-CAM is computed on the last residual block (layer4) of
ResNet18, producing a heatmap that highlights the image regions that most
influenced the predicted class — e.g. it should highlight the hole for
"Circle" or the scratch line for "Line".
"""},

{
"id": "model_performance",
"title": "Model Performance — 5-Fold Cross-Validation Results",
"content": """
The model was evaluated with stratified 5-fold cross-validation (random_state
fixed for reproducibility): each fold trains on 120 images and tests on the
remaining 30 (10 per class), and a fresh ResNet18 head is trained from scratch
for each fold.

Reported results (see Model Performance page for live numbers from the most
recent training run):
- Average accuracy across 5 folds: ~0.94 (94%)
- Per-fold accuracy ranges from ~0.83 to 1.00 — i.e. some folds reach perfect
  accuracy on their 30-image test set, while others misclassify a handful of
  images.
- The confusion matrix shows confusion is concentrated between "Circle" and
  "No defect" — a small/faint hole can resemble a normal weave gap if lighting
  is uneven.

How to read the fold variance: with only 30 test images per fold, ONE
misclassified image already changes accuracy by ~3.3 percentage points. The
83%-100% spread is therefore expected statistical noise for a dataset of this
size, not evidence of an unstable model — but it does mean the 94% average
should be reported with its standard deviation, not as a single point
estimate, and that more data would tighten this estimate considerably.
"""},

{
"id": "sop_circle_defect",
"title": "SOP — Circle Defect (Hole / Puncture)",
"content": """
Defect description: a roughly circular hole or puncture in the checkered
weave, where one or more yarns are missing or torn, exposing a gap in the
fabric.

Likely root causes, in order of frequency:
1. Yarn breakage / pull-out: a single weft or warp yarn breaks during weaving
   and is pulled out by the loom's reciprocating motion, leaving a hole.
   Often caused by excessive yarn tension or a weak/spliced yarn section.
2. Spark or ember burn: a hot particle (e.g. from a nearby cutting or
   finishing process) lands on the fabric and burns through one or more
   threads, leaving a small round scorch hole.
3. Foreign object puncture: a sharp protrusion on a guide roller, reed, or
   heald wire snags and tears the fabric as it passes.
4. Pest damage: moth or insect damage to stored yarn/fabric, visible as small
   round holes, often with frayed edges.

Corrective actions:
1. Stop the line and isolate the affected roll segment (mark meterage /
   roll position for traceability).
2. Inspect the loom's warp and weft yarn paths for broken or low-tension
   yarns near the defect location; re-thread or re-tension as needed.
3. Check upstream rollers, reed, and heald wires for burrs or sharp edges
   using a cloth-wipe test (a snag on cloth indicates a burr).
4. If burn marks are present, inspect nearby hot processes (heaters, cutting
   stations) for stray sparks and check for adequate shielding.
5. Log the defect (type=Circle, roll ID, meterage, timestamp) for SPC
   tracking; if Circle-defect rate exceeds the control limit, escalate to
   the shift supervisor for a loom inspection.

Severity: typically a MAJOR defect (a hole is a functional failure of the
fabric, not just cosmetic) — see the Quality Standard Reference document for
point-based grading.
"""},

{
"id": "sop_line_defect",
"title": "SOP — Line Defect (Scratch / Cut Mark)",
"content": """
Defect description: a linear mark crossing the checkered pattern — visually,
a straight or near-straight scratch, cut, or abrasion line that disrupts the
regular grid of the weave.

Likely root causes, in order of frequency:
1. Sharp burr on a guide roller or reed: as the fabric travels under tension
   over a roller with a nicked or burred edge, the surface yarns are
   abraded or cut in a straight line along the direction of travel.
2. Misaligned trimming/cutting station: a cutting blade positioned slightly
   too low scores the fabric surface without fully separating it.
3. Snagged foreign yarn or thread end dragging across the web, scoring a
   line as the fabric advances.
4. Static-induced yarn snag: electrostatic charge causes loose fibres to
   catch on machine parts and drag, leaving a faint line defect.

Corrective actions:
1. Stop the line and isolate the affected roll segment (mark meterage / roll
   position for traceability).
2. Run a cloth-wipe or hand-pass test along all rollers and the reed in the
   zone upstream of the defect to locate burrs or sharp edges; deburr or
   replace the component.
3. Verify cutting/trimming station blade height and alignment against the
   setup specification; recalibrate if out of tolerance.
4. Check for loose threads/fibres on guide elements and clean as part of
   autonomous maintenance (5S / daily cleaning checklist).
5. If static is suspected, verify anti-static bar/ioniser operation near the
   inspection frame.
6. Log the defect (type=Line, roll ID, meterage, timestamp) for SPC tracking;
   recurring Line defects at a similar roll position across multiple rolls
   point to a fixed mechanical cause (a specific roller/blade) rather than a
   yarn-quality issue.

Severity: typically a MAJOR defect if the weave is cut/abraded through
(structural weak point); MINOR if only surface fibres are disturbed without
yarn breakage — visual inspection of the Grad-CAM-highlighted region together
with a manual hand-feel check determines which applies.
"""},

{
"id": "sop_no_defect",
"title": "SOP — \"No Defect\" Classification and False-Negative Risk",
"content": """
A "No defect" prediction means the inspected patch shows a regular, intact
checkered pattern: continuous warp and weft yarns, consistent check size and
alignment, no holes, cuts, or foreign marks.

Acceptance criteria for a patch to be genuinely defect-free:
- Check (square) size and grid alignment consistent with the reference
  pattern.
- No visible breaks in warp or weft yarns.
- No discoloration, stains, or foreign material on the surface.

False-negative risk (the most safety-critical failure mode for a quality
system): a real defect classified as "No defect" passes inspection
undetected. Based on the 5-fold cross-validation confusion matrix, the
main confusion observed is between "Circle" and "No defect" — small or
faint holes near the edge of a check can be visually subtle, especially
under uneven lighting.

Mitigations for false negatives in a real deployment:
1. Use a confidence threshold: if the top-class probability for "No defect"
   is below a set threshold (e.g. 0.8) and the second-best class is "Circle"
   or "Line", flag the frame for human review rather than auto-passing it.
2. Standardise inspection-frame lighting (diffuse, consistent illumination)
   to reduce the visual ambiguity that drives this confusion.
3. Periodically audit a sample of "No defect" frames manually and feed any
   missed defects back into the training set (active learning loop).
"""},

{
"id": "quality_standard_reference",
"title": "Quality Standard Reference — 4-Point Fabric Inspection System",
"content": """
Textile mills commonly grade fabric quality using point-grading systems such
as the 4-Point (American) or 10-Point system (related to ASTM D5430,
"Standard Test Methods for Visually Inspecting and Grading Fabrics"). These
systems assign penalty points to defects based on their length/size, and
limit the total points per 100 square yards (or per defined roll length) that
a roll may have to be classed as "first quality".

Typical 4-point system penalty bands (illustrative):
- Defect up to 3 inches: 1 point
- Defect 3-6 inches: 2 points
- Defect 6-9 inches: 3 points
- Defect over 9 inches: 4 points

How the AOI digital twin's output maps onto this scheme:
- "Circle" defects (holes/punctures) are typically scored at the higher end
  of the penalty band regardless of size, because a hole is a structural
  defect, not merely cosmetic.
- "Line" defects are scored based on their length (the AOI bounding box /
  Grad-CAM extent gives an estimate of defect length in pixels, which can be
  converted to physical length using the known field-of-view).
- Rolls exceeding the points-per-100-square-yards limit are downgraded to
  "seconds" (lower-value product) or rejected, depending on the buyer's
  quality agreement.

In a full deployment, the digital twin's per-frame classifications would be
aggregated per roll to compute a running point total and an automatic
first-quality / seconds / reject classification at the end of each roll.
"""},

{
"id": "aoi_maintenance",
"title": "AOI Station Maintenance and Calibration Guide",
"content": """
The Automated Optical Inspection (AOI) station's prediction quality depends
on consistent image acquisition. Maintenance and calibration tasks:

Daily (autonomous maintenance / operator level):
- Clean the camera lens and inspection-frame glass/cover of lint and dust
  (fabric production generates significant airborne fibre).
- Visual check that lighting units are all functional (no dim/failed LEDs) —
  uneven lighting is the leading cause of false "No defect" / "Circle"
  confusion (see SOP — No Defect document).
- Check fabric tension and speed at the inspection frame are within the
  setpoint range; motion blur from excessive speed degrades image quality.

Weekly (technician level):
- Re-run a calibration image set (known good + known defective samples)
  through the model and compare predictions to the 5-fold CV baseline
  metrics; a significant accuracy drop indicates camera drift, lighting
  degradation, or a fabric pattern change requiring retraining.
- Verify camera focus and field-of-view alignment against reference marks.

Periodic (engineering level, e.g. on fabric pattern change or after
significant accuracy drift):
- Collect new labelled images for any new fabric pattern or defect type and
  retrain/fine-tune the classifier (the training pipeline in scripts/train.py
  can be re-run with an updated dataset/ folder).
- Re-run the 5-fold cross-validation to confirm the retrained model meets or
  exceeds the previous accuracy baseline before deploying it to the line.
"""},

{
"id": "deployment_integration",
"title": "Deployment and Integration in a Real Industrial System",
"content": """
This prototype reads static images from a local dataset/ folder and runs
inference on demand from the Streamlit dashboard. A real deployment would
differ as follows:

Data flow:
1. AOI camera (DCE) streams frames continuously as the fabric web moves.
2. A frame-grabber / edge device runs the same pre-processing + ResNet18
   inference pipeline (core/model.py) in real time, e.g. at the line speed
   (frames per second matched to fabric throughput).
3. Each prediction (class, confidence, Grad-CAM region, roll ID, meterage,
   timestamp) is written to the MES/QMS (Cross System Entity in ISO 23247
   terms) for traceability and SPC charting.
4. If a defect is detected above a confidence threshold, the system can:
   - raise an operator alert on the User Entity dashboard,
   - trigger a physical marker/flag on the fabric roll at that meterage,
   - (in a fully automated line) trigger a reject/diverter mechanism.
5. The Quality Assistant (RAG) would be available on a tablet/HMI at the
   inspection frame so operators can immediately query "what does a Circle
   defect mean and what should I check?" without leaving the line.

Integration challenges to discuss critically:
- Real-time throughput: inference must keep up with line speed; ResNet18 on a
  frozen backbone is lightweight enough for this on modest edge hardware/GPU.
- Domain shift: the prototype's "Circle"/"Line" defects were created by
  placing objects on good fabric, which differs visually from naturally
  occurring defects (e.g. an actual yarn-pullout hole has frayed edges a
  clean punched hole does not). A pilot phase with real defective rolls and
  retraining would be required before full deployment.
- Traceability: linking a defect event to a specific roll/meterage/loom
  requires synchronising the AOI's frame timestamps with the line's
  encoder/meterage counter — not implemented in this prototype.
- Alert fatigue: thresholds must be tuned so operators are not overwhelmed by
  low-confidence "maybe defect" alerts; the confidence-threshold approach in
  the No Defect SOP applies here too.
"""},

{
"id": "limitations_discussion",
"title": "Critical Discussion — Dataset and Model Limitations",
"content": """
Honest limitations of this prototype, to be addressed explicitly in the
project report's critical-discussion section:

1. Small dataset (N=150, 50 per class): 5-fold CV gives an average accuracy
   of ~94% but with notable per-fold variance (83%-100%), reflecting the
   small test-set size (30 images/fold) rather than a precisely-known true
   accuracy. A production system would need hundreds to thousands of labelled
   images per class.

2. Synthetic / staged defects: "Circle" and "Line" defects were created by
   placing physical objects (a punched hole, a ruler/blade) on otherwise good
   fabric, rather than being naturally occurring weaving defects. This is a
   reasonable and common approach for a small student prototype, but the
   visual signatures of staged vs. real defects can differ (e.g. edge
   sharpness, lighting/shadow artefacts from the placed object itself, which
   the model may partly be learning instead of the "defect" per se). Grad-CAM
   visualisations should be checked to confirm the model focuses on the
   defect region and not on shadow/lighting artefacts from how the image was
   staged.

3. Single fabric pattern: the model is trained only on this blue/yellow
   checkered pattern. Generalisation to other check sizes/colours or other
   weave types is untested and likely requires retraining.

4. No temporal/positional context: each frame is classified independently;
   a real system would also track defect frequency over roll length (for SPC)
   and correlate with loom/machine settings (for root-cause analysis) — this
   prototype demonstrates per-frame classification only, which is the
   "quality monitoring" building block, not the full SPC/root-cause layer.

5. RAG knowledge base is a prototype corpus: the SOP documents in this
   Quality Assistant are illustrative, written for this project rather than
   sourced from a real mill's quality manuals. In a real deployment the
   corpus would be the mill's actual SOPs, maintenance manuals, and the
   relevant textile quality standards (e.g. the full ASTM D5430 text).
"""},

]  # end of _DOCS


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

class RAGKnowledgeBase:
    """TF-IDF retrieval with optional sentence-transformer upgrade."""

    def __init__(self):
        self._docs = _DOCS
        self._texts = [d["title"] + "\n" + d["content"] for d in _DOCS]
        self._tfidf_matrix = None
        self._vocab = {}
        self._semantic_embeddings = None
        self._build_tfidf()
        if _HAS_SEMANTIC:
            self._build_semantic()

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_tfidf(self):
        corpus = self._texts
        tokenised = [self._tokenise(t) for t in corpus]
        vocab = sorted({tok for doc in tokenised for tok in doc})
        self._vocab = {w: i for i, w in enumerate(vocab)}
        n, V = len(corpus), len(vocab)

        tf = np.zeros((n, V), dtype=np.float32)
        for d, tokens in enumerate(tokenised):
            for tok in tokens:
                if tok in self._vocab:
                    tf[d, self._vocab[tok]] += 1
            s = tf[d].sum()
            if s > 0:
                tf[d] /= s

        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((n + 1) / (df + 1)) + 1.0
        self._tfidf_matrix = tf * idf

        norms = np.linalg.norm(self._tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._tfidf_matrix /= norms

    def _build_semantic(self):
        try:
            model = _get_semantic_model()
            self._semantic_embeddings = model.encode(
                self._texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            self._semantic_embeddings = None

    def _tfidf_query(self, query: str) -> np.ndarray:
        tokens = self._tokenise(query)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return self._tfidf_matrix @ vec

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        if _HAS_SEMANTIC and self._semantic_embeddings is not None:
            try:
                model = _get_semantic_model()
                q_emb = model.encode([query], normalize_embeddings=True)[0]
                scores = self._semantic_embeddings @ q_emb
            except Exception:
                scores = self._tfidf_query(query)
        else:
            scores = self._tfidf_query(query)

        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._docs[i]) for i in top_idx]


# ── Local answer composition (no LLM) ────────────────────────────────────────

def compose_local_answer(query: str, results: List[Tuple[float, dict]], kb: RAGKnowledgeBase) -> str:
    """Compose an answer from retrieved documents without any LLM."""
    if not results:
        return "No relevant information found in the knowledge base."

    best_score, best_doc = results[0]
    query_toks = set(kb._tokenise(query))

    candidate_sentences = []
    for _, doc in results[:2]:
        for line in doc["content"].split("\n"):
            line = line.strip()
            if len(line) < 30:
                continue
            line_toks = set(kb._tokenise(line))
            overlap = len(query_toks & line_toks)
            candidate_sentences.append((overlap, line, doc["title"]))

    candidate_sentences.sort(reverse=True)

    seen, answer_parts = set(), []
    answer_parts.append(f"**{best_doc['title']}**\n")

    for _, sent, _ in candidate_sentences[:6]:
        if sent not in seen and len(sent) > 40:
            answer_parts.append(sent)
            seen.add(sent)

    if len(answer_parts) == 1:
        return f"**{best_doc['title']}**\n\n{best_doc['content'].strip()[:600]}"

    return "\n\n".join(answer_parts)


# ── Claude API generation (optional upgrade) ─────────────────────────────────

def _get_api_key() -> "str | None":
    def _valid(k: "str | None") -> "str | None":
        return k if (k and k.startswith("sk-ant-") and len(k) > 20) else None

    key = _valid(os.environ.get("ANTHROPIC_API_KEY"))
    if key:
        return key
    try:
        import streamlit as st
        return _valid(st.secrets.get("ANTHROPIC_API_KEY"))
    except Exception:
        return None


def generate_answer(query: str, context_chunks: List[str]) -> "str | None":
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    context = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are a quality assistant for an Automated Optical Inspection (AOI) "
        "digital twin on a checkered-fabric weaving line. Answer the operator's "
        "question concisely and practically, grounded ONLY in the provided "
        "context. If the context does not contain the answer, say so explicitly "
        "rather than guessing."
    )
    prompt = f"Context:\n\n{context}\n\nQuestion: {query}"
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except anthropic.AuthenticationError:
        return None
    except Exception:
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_kb: "RAGKnowledgeBase | None" = None


def get_knowledge_base() -> RAGKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = RAGKnowledgeBase()
    return _kb
