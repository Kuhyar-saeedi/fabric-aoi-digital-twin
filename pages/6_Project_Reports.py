"""
pages/6_Project_Reports.py
===========================
Project deliverables (final report + presentation slides) viewable and
downloadable directly from the dashboard, so the project can be presented
online without any local files.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Project Reports — Checkered Fabric AOI", page_icon="📄", layout="wide")

ROOT = Path(__file__).resolve().parents[1]

st.title("📄 Project Reports & Slides")
st.caption(
    "Final written report and presentation deck for the Smart Factories course project "
    "work — view inline or download below."
)

DOCS = {
    "📄 Final Report": ROOT / "final_report.pdf",
    "📊 Presentation Slides": ROOT / "Fabric_Quality_Digital_Twin.pdf",
}


@st.cache_data
def load_pdf(path: str) -> bytes:
    return Path(path).read_bytes()


tabs = st.tabs(list(DOCS.keys()))

for tab, (label, path) in zip(tabs, DOCS.items()):
    with tab:
        if not path.exists():
            st.error(f"`{path.name}` not found in the repository.")
            continue

        data = load_pdf(str(path))
        st.download_button(
            f"⬇️ Download {path.name}",
            data=data,
            file_name=path.name,
            mime="application/pdf",
            key=f"dl_{path.name}",
        )

        b64 = base64.b64encode(data).decode()
        components.html(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="900" style="border:1px solid #334155; border-radius:8px;">'
            f"</iframe>",
            height=920,
        )
