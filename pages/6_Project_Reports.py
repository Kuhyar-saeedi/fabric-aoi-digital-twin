"""
pages/6_Project_Reports.py
===========================
Project deliverables (final report + presentation slides) viewable and
downloadable directly from the dashboard, so the project can be presented
online without any local files.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from core.i18n import lang_selector, t

st.set_page_config(page_title="Project Reports — Checkered Fabric AOI", page_icon="📄", layout="wide")

lang_selector()

ROOT = Path(__file__).resolve().parents[1]

st.title(t("rep_title"))
st.caption(t("rep_caption"))

DOCS = {
    t("rep_tab_report"): ROOT / "final_report.pdf",
    t("rep_tab_slides"): ROOT / "Fabric_Quality_Digital_Twin.pdf",
}


@st.cache_data
def load_pdf(path: str) -> bytes:
    return Path(path).read_bytes()


tabs = st.tabs(list(DOCS.keys()))

for tab, (label, path) in zip(tabs, DOCS.items()):
    with tab:
        if not path.exists():
            st.error(t("rep_not_found", name=path.name))
            continue

        data = load_pdf(str(path))
        st.download_button(
            t("rep_download", name=path.name),
            data=data,
            file_name=path.name,
            mime="application/pdf",
            key=f"dl_{path.name}",
        )

        pdf_viewer(input=data, width="100%", height=1100)
