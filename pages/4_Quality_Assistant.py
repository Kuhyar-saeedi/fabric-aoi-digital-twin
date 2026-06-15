"""
pages/4_Quality_Assistant.py
=============================
RAG-based Quality Assistant ("Track 3 — RAG for Industrial Knowledge
Management"). Lets an operator ask natural-language questions about defect
causes, corrective actions, the ISO 23247 mapping, and quality standards.
Answers are grounded in (and cite) a small SOP knowledge base.
"""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.rag import compose_local_answer, generate_answer, get_knowledge_base  # noqa: E402
from core.i18n import get_lang, lang_selector, t  # noqa: E402

st.set_page_config(page_title="Quality Assistant — Checkered Fabric AOI", page_icon="💬", layout="wide")

lang_selector()

st.title(t("qa_title"))
st.caption(t("qa_caption"))

lang = get_lang()
kb = get_knowledge_base(lang)

import core.rag as ragmod
has_claude = ragmod._get_api_key() is not None
if has_claude:
    st.success(t("qa_claude_success"), icon="🤖")
else:
    st.info(t("qa_claude_info"), icon="ℹ️")

EXAMPLES = [
    t("qa_ex1"),
    t("qa_ex2"),
    t("qa_ex3"),
    t("qa_ex4"),
    t("qa_ex5"),
]

st.markdown(t("qa_try_example"))
cols = st.columns(len(EXAMPLES))
clicked = None
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        clicked = ex

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

query = st.chat_input(t("qa_chat_placeholder")) or clicked

if query:
    results = kb.retrieve(query, top_k=3)
    answer = None
    if has_claude:
        answer = generate_answer(query, [d["content"] for _, d in results], lang)
    if answer is None:
        answer = compose_local_answer(query, results, kb)
    st.session_state.qa_history.append((query, answer, results))

for query, answer, results in reversed(st.session_state.qa_history):
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander(t("qa_sources", n=len(results))):
            for score, doc in results:
                st.markdown(t("qa_source_relevance", title=doc["title"], score=f"{score:.2f}"))
                st.caption(doc["content"].strip()[:300] + "...")

if not st.session_state.qa_history:
    st.markdown("---")
    with st.expander(t("qa_kb_expander")):
        for doc in kb._docs:
            st.markdown(f"- **{doc['title']}**")
