"""
pages/4_Quality_Assistant.py
=============================
RAG-based Quality Assistant ("Track 3 — RAG for Industrial Knowledge
Management"). Lets an operator ask natural-language questions about defect
causes, corrective actions, the ISO 23247 mapping, and quality standards.
Answers are grounded in (and cite) a small SOP knowledge base.

Voice input:  browser Web Speech API (Chrome/Edge, free, no key)
Voice output: browser speechSynthesis (all modern browsers, offline)
"""

from pathlib import Path
import re
import sys

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.rag import compose_local_answer, generate_answer, get_knowledge_base  # noqa: E402
from core.i18n import get_lang, lang_selector, t  # noqa: E402

st.set_page_config(page_title="Quality Assistant — Checkered Fabric AOI", page_icon="💬", layout="wide")

lang_selector()


# ── Voice helpers: browser STT (Web Speech API) + TTS (speechSynthesis) ───────
def _tts_html(text: str, answer_id: int, lang: str = "en") -> str:
    """Height-0 iframe that speaks `text` once per answer via speechSynthesis."""
    safe = (text.replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("$", "\\$")
                .replace("\n", " ")
                .replace('"', '\\"'))
    return f"""
<script>
(function(){{
  const id={answer_id};
  const prev=parseInt(sessionStorage.getItem('qa_tts_id')||'0');
  if(id<=prev)return;
  sessionStorage.setItem('qa_tts_id',String(id));
  if(!('speechSynthesis' in window))return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(`{safe}`);
  u.rate=1.05;u.pitch=1.0;u.volume=0.92;
  const voices=window.speechSynthesis.getVoices();
  const pref=voices.find(v=>v.lang.startsWith('{lang}')&&v.localService)
            ||voices.find(v=>v.lang.startsWith('{lang}'));
  if(pref)u.voice=pref;
  window.speechSynthesis.speak(u);
}})();
</script>
"""


def _stop_speaking_html(label: str) -> str:
    return f"""
<button onclick="window.parent.speechSynthesis.cancel()" style="
    display:inline-block;padding:6px 14px;margin-top:1px;
    background:#3a1a1a;border:1px solid #8B2222;border-radius:6px;
    color:#ff6b6b;font-size:12px;font-weight:600;cursor:pointer;
    font-family:inherit;white-space:nowrap;transition:background .2s;"
    onmouseover="this.style.background='#5a2020'"
    onmouseout="this.style.background='#3a1a1a'">
    {label}
</button>
"""


# Voice-input mic button (Web Speech API). The recognized transcript is
# written into a hidden st.text_input (matched by aria-label) so Python can
# read it on the next rerun -- avoids relying on st.chat_input internals.
_VOICE_HIDDEN_LABEL = "qa_voice_raw_input"


def _voice_html(stt_lang: str, voice_btn_label: str) -> str:
    return f"""
<div style="display:flex;align-items:center;gap:8px;">
  <button id="micBtn" style="
      background:#252836;border:1px solid #3a3d4e;border-radius:6px;
      padding:6px 14px;margin-top:1px;font-size:12px;color:#ccc;cursor:pointer;
      white-space:nowrap;font-family:inherit;font-weight:600;transition:all .2s;">
    {voice_btn_label}
  </button>
  <span id="micStatus" style="font-size:11px;color:#888;"></span>
</div>
<script>
(function(){{
  let _rec=null, _on=false;
  const btn=document.getElementById('micBtn');
  const status=document.getElementById('micStatus');

  btn.onclick=()=>{{_on?stopMic():startMic();}};

  function startMic(){{
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){{status.textContent='No STT - use Chrome/Edge';return;}}
    _rec=new SR();
    _rec.lang='{stt_lang}';
    _rec.continuous=false;
    _rec.interimResults=true;
    _rec.onstart=()=>{{
      _on=true;
      btn.style.background='#4CAF50';btn.style.color='#000';btn.style.borderColor='#4CAF50';
      status.textContent='Listening...';
    }};
    _rec.onresult=(e)=>{{
      let text='';
      for(let i=e.resultIndex;i<e.results.length;i++)text+=e.results[i][0].transcript;
      const preview=text.length>40?text.substring(0,40)+'...':text;
      status.textContent='>> '+preview;
      if(e.results[e.results.length-1].isFinal)submitQ(text);
    }};
    _rec.onerror=(e)=>{{status.textContent='Error: '+e.error;resetBtn();}};
    _rec.onend=resetBtn;
    _rec.start();
  }}
  function stopMic(){{if(_rec)_rec.stop();}}
  function resetBtn(){{
    _on=false;
    btn.style.background='#252836';btn.style.color='#ccc';btn.style.borderColor='#3a3d4e';
  }}
  function submitQ(text){{
    const inp=window.parent.document.querySelector('input[aria-label="{_VOICE_HIDDEN_LABEL}"]');
    if(!inp){{status.textContent='Failed: no target';return;}}
    const setter=Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype,'value').set;
    setter.call(inp,text);
    inp.dispatchEvent(new Event('input',{{bubbles:true}}));
    setTimeout(()=>{{
      inp.dispatchEvent(new KeyboardEvent('keydown',{{key:'Enter',keyCode:13,bubbles:true,cancelable:true}}));
      status.textContent='Sent!';
      setTimeout(()=>{{status.textContent='';}},2000);
    }},80);
  }}
}})();
</script>
"""


st.title(t("qa_title"))
st.caption(t("qa_caption"))

lang = get_lang()
kb = get_knowledge_base(lang)

# ── Sidebar: knowledge-base reference ───────────────────────────────────────
with st.sidebar:
    st.markdown(f"#### {t('qa_kb_sidebar_hdr')}")
    with st.expander(t("qa_kb_expander"), expanded=False):
        for doc in kb._docs:
            st.markdown(f"- **{doc['title']}**")

import core.rag as ragmod
has_claude = ragmod._get_api_key() is not None
if has_claude:
    st.success(t("qa_claude_success"), icon="🤖")
else:
    st.info(t("qa_claude_info"), icon="ℹ️")

for _k, _v in [
    ("qa_history", []),
    ("qa_tts_enabled", True),
    ("qa_answer_tts", ""),
    ("qa_answer_counter", 0),
    ("qa_last_voice_transcript", ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

EXAMPLES = [
    t("qa_ex1"),
    t("qa_ex2"),
    t("qa_ex3"),
    t("qa_ex4"),
    t("qa_ex5"),
]

st.divider()
st.markdown(f"### {t('qa_chat_hdr')}")
st.markdown(t("qa_try_example"))
cols = st.columns(len(EXAMPLES))
clicked = None
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        clicked = ex

st.write("")

# ── Chat history ─────────────────────────────────────────────────────────────
for hist_query, hist_answer, hist_results in reversed(st.session_state.qa_history):
    with st.chat_message("user"):
        st.markdown(hist_query)
    with st.chat_message("assistant"):
        st.markdown(hist_answer)
        with st.expander(t("qa_sources", n=len(hist_results))):
            for score, doc in hist_results:
                st.markdown(t("qa_source_relevance", title=doc["title"], score=f"{score:.2f}"))
                st.caption(doc["content"].strip()[:300] + "...")

# TTS playback — inject a 0-height iframe that speaks the latest answer
if st.session_state.qa_tts_enabled and st.session_state.qa_answer_tts:
    components.html(
        _tts_html(st.session_state.qa_answer_tts, st.session_state.qa_answer_counter, lang),
        height=0,
    )

# ── Voice & audio controls (bottom of page, just above the chat input) ────────
stt_lang = "it-IT" if lang == "it" else "en-US"

# Hidden text input used as a bridge: the mic button's JS writes the
# recognized transcript here, matched by its aria-label.
st.text_input(_VOICE_HIDDEN_LABEL, key="qa_voice_raw", label_visibility="collapsed")
st.markdown(
    f"""
    <style>
    div[data-testid="stTextInput"]:has(input[aria-label="{_VOICE_HIDDEN_LABEL}"]) {{
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        opacity: 0;
        pointer-events: none;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(f"**{t('qa_voice_hdr')}**")
    st.caption(t("qa_voice_caption"))
    ctrl_cols = st.columns([1, 1.6, 1.6, 5])
    with ctrl_cols[0]:
        components.html(_voice_html(stt_lang, t("qa_voice_btn")), height=40)
    with ctrl_cols[1]:
        st.session_state.qa_tts_enabled = st.toggle(
            t("qa_tts_toggle"), value=st.session_state.qa_tts_enabled
        )
    with ctrl_cols[2]:
        if st.session_state.qa_tts_enabled:
            components.html(_stop_speaking_html(t("qa_stop_speaking")), height=40)

voice_query = None
_voice_raw = st.session_state.get("qa_voice_raw", "")
if _voice_raw and _voice_raw != st.session_state.qa_last_voice_transcript:
    st.session_state.qa_last_voice_transcript = _voice_raw
    voice_query = _voice_raw

chat_query = st.chat_input(t("qa_chat_placeholder"))

query = chat_query or clicked or voice_query

if query:
    with st.spinner(t("qa_thinking")):
        results = kb.retrieve(query, top_k=3)
        answer = None
        if has_claude:
            answer = generate_answer(query, [d["content"] for _, d in results], lang)
        if answer is None:
            answer = compose_local_answer(query, results, kb)
    st.session_state.qa_history.append((query, answer, results))

    _clean = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    _clean = re.sub(r"[#*`]", "", _clean).strip()
    st.session_state.qa_answer_tts = _clean[:400]
    st.session_state.qa_answer_counter += 1
    st.rerun()
