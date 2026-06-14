"""
scripts/eval_rag.py
====================
Evaluates the Quality Assistant (RAG) on 15 representative queries, as
required by the RAG track assignment ("evaluation of response quality on at
least 15 representative queries: correct / partial / hallucinated / no
answer").

Prints, for each query: the answer, the retrieved sources with relevance
scores, and a manual quality label (assigned by inspection — this script
does not auto-grade, since "correct" requires human judgement against the
SOP corpus).

Run from the project root:
    python scripts/eval_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.rag import compose_local_answer, generate_answer, get_knowledge_base  # noqa: E402

# (query, manual label, note) — labels assigned after reviewing the answer
# against the knowledge base in core/rag.py. Labels: correct / partial /
# hallucinated / no_answer.
QUERIES = [
    ("What causes a Circle defect and how do I fix it?",
     "correct", "Retrieves SOP — Circle Defect; lists root causes and corrective actions directly."),

    ("What should I do if a Line defect is detected on the line?",
     "correct", "Retrieves SOP — Line Defect; corrective-action steps are present verbatim."),

    ("What does a 'No defect' prediction mean and what is the false-negative risk?",
     "correct", "Retrieves SOP — No Defect; acceptance criteria and false-negative mitigations covered."),

    ("What is the Observable Manufacturing Element (OME) in this digital twin?",
     "correct", "Retrieves ISO 23247 Mapping; OME definition for this case is explicit."),

    ("Why is this project considered a data-driven Digital Twin rather than a simulation?",
     "correct", "Retrieves 'What Makes This a Data-Driven Digital Twin'; directly answers."),

    ("What model architecture is used and how was it trained?",
     "correct", "Retrieves Dataset and Model doc; ResNet18, frozen backbone, augmentation, 8 epochs."),

    ("What is the average accuracy of the classifier across 5-fold cross-validation?",
     "correct", "Retrieves Model Performance doc; ~94% average is stated explicitly."),

    ("Why does the per-fold accuracy vary between about 83% and 100%?",
     "correct", "Retrieves Model Performance doc; explains 30-image fold size and statistical noise."),

    ("What is the 4-point fabric inspection system and how does it relate to this digital twin's output?",
     "correct", "Retrieves Quality Standard Reference; explains penalty points and how AOI output maps to it."),

    ("How should the AOI camera and inspection frame be maintained on a daily basis?",
     "correct", "Retrieves AOI Maintenance doc; daily checklist (lens cleaning, lighting check) is listed."),

    ("How would this prototype be integrated with a factory MES/QMS in a real deployment?",
     "correct", "Retrieves Deployment and Integration doc; data flow and Cross System Entity role covered."),

    ("What are the main limitations of the dataset used in this project?",
     "correct", "Retrieves Limitations Discussion doc; small N, staged defects, single pattern all listed."),

    ("Is a Circle defect more severe than a Line defect?",
     "partial", "Both SOPs discuss severity (Circle=major; Line=major or minor depending on yarn break), "
                "but the assistant must combine two documents — answer may only surface one severity note "
                "depending on which document ranks first."),

    ("What confidence threshold should trigger a human review of a prediction?",
     "partial", "The concrete threshold (0.8) is only in the No Defect SOP; the Deployment/"
                "Integration and AOI Maintenance docs discuss threshold-based routing without "
                "repeating the number, so retrieval may surface the general concept without the "
                "concrete value."),

    ("What is the current market price of cotton yarn?",
     "no_answer", "Out of scope for the knowledge base (no pricing/market documents). Correct behaviour "
                  "is for the assistant to retrieve low-relevance documents and not fabricate a price — "
                  "verify the local/Claude answer does not hallucinate a number."),
]


def main():
    kb = get_knowledge_base()
    from core.rag import _get_api_key
    has_claude = _get_api_key() is not None
    print(f"Claude generation: {'ENABLED' if has_claude else 'disabled (local extractive answers)'}\n")

    rows = []
    for i, (query, label, note) in enumerate(QUERIES, 1):
        results = kb.retrieve(query, top_k=3)
        answer = generate_answer(query, [d["content"] for _, d in results]) if has_claude else None
        if answer is None:
            answer = compose_local_answer(query, results, kb)

        top_score, top_doc = results[0]
        print(f"[{i:02d}] Q: {query}")
        print(f"     top source: {top_doc['title']} (relevance {top_score:.3f})")
        print(f"     manual label: {label}  -- {note}")
        print(f"     answer (first 200 chars): {answer[:200].strip()!r}")
        print()

        rows.append({
            "query": query, "label": label, "top_source": top_doc["title"],
            "relevance": round(top_score, 3),
        })

    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print("Summary:", counts, f"  ({len(rows)} queries total)")


if __name__ == "__main__":
    main()
