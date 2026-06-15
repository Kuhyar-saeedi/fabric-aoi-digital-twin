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
    width:100%;padding:7px 0;margin-top:4px;
    background:#3a1a1a;border:1px solid #8B2222;border-radius:6px;
    color:#ff6b6b;font-size:13px;font-weight:600;cursor:pointer;
    font-family:inherit;transition:background .2s;"
    onmouseover="this.style.background='#5a2020'"
    onmouseout="this.style.background='#3a1a1a'">
    {label}
</button>
"""


_VOICE_HTML_TMPL = (
    '<div style="display:flex;align-items:center;gap:10px;padding:2px 0;">'
    '  <button id="micBtn" onclick="toggleMic()" style="'
    '    background:#252836;border:1px solid #3a3d4e;border-radius:6px;'
    '    padding:7px 16px;font-size:13px;color:#ccc;cursor:pointer;'
    '    transition:all .2s;white-space:nowrap;font-family:inherit;">'
    '    __VOICE_BTN__'
    '  </button>'
    '  <span id="micStatus" style="font-size:11px;color:#777;flex:1;"></span>'
    '</div>'
    '<script>'
    'let _rec=null,_on=false;'
    'function toggleMic(){_on?stopMic():startMic();}'
    'function startMic(){'
    '  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;'
    "  if(!SR){setS('No STT — use Chrome/Edge');return;}"
    "  _rec=new SR();_rec.lang='__STT_LANG__';_rec.continuous=false;_rec.interimResults=true;"
    '  _rec.onstart=()=>{'
    '    _on=true;'
    "    const b=document.getElementById('micBtn');"
    "    b.textContent='Stop';"
    "    b.style.cssText+='background:#4CAF50;color:#000;border-color:#4CAF50;';"
    "    setS('Listening...');"
    '  };'
    '  _rec.onresult=(e)=>{'
    "    let t='';"
    '    for(let i=e.resultIndex;i<e.results.length;i++)t+=e.results[i][0].transcript;'
    "    const preview=t.length>55?t.substring(0,55)+'...':t;"
    "    setS('>> '+preview);"
    '    if(e.results[e.results.length-1].isFinal)submitQ(t);'
    '  };'
    "  _rec.onerror=(e)=>{setS('Error: '+e.error);resetBtn();};"
    '  _rec.onend=resetBtn;'
    '  _rec.start();'
    '}'
    'function stopMic(){if(_rec)_rec.stop();}'
    'function resetBtn(){'
    '  _on=false;'
    "  const b=document.getElementById('micBtn');"
    "  b.textContent='__VOICE_BTN__';"
    "  b.style.background='#252836';b.style.color='#ccc';b.style.borderColor='#3a3d4e';"
    '}'
    "function setS(s){const el=document.getElementById('micStatus');if(el)el.textContent=s;}"
    'function submitQ(text){'
    '  const ta=window.parent.document.querySelector('
    "    'textarea[data-testid=\"stChatInputTextArea\"]');"
    "  if(!ta){setS('Failed: '+text);return;}"
    '  const setter=Object.getOwnPropertyDescriptor('
    "    window.parent.HTMLTextAreaElement.prototype,'value').set;"
    '  setter.call(ta,text);'
    "  ta.dispatchEvent(new Event('input',{bubbles:true}));"
    '  setTimeout(()=>{'
    "    ta.dispatchEvent(new KeyboardEvent('keydown',"
    "      {key:'Enter',keyCode:13,bubbles:true,cancelable:true}));"
    "    setS('Submitted!');"
    "    setTimeout(()=>setS(''),2500);"
    '  },130);'
    '}'
    '</script>'
)


def _voice_html(lang: str = "en") -> str:
    stt_lang = "it-IT" if lang == "it" else "en-US"
    voice_btn = t("qa_voice_btn")
    return _VOICE_HTML_TMPL.replace("__STT_LANG__", stt_lang).replace("__VOICE_BTN__", voice_btn)

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

for _k, _v in [
    ("qa_history", []),
    ("qa_tts_enabled", True),
    ("qa_answer_tts", ""),
    ("qa_answer_counter", 0),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Voice controls ─────────────────────────────────────────────────────────────
st.divider()
st.caption(t("qa_voice_caption"))
voice_col, tts_col = st.columns([2, 1])
with voice_col:
    components.html(_voice_html(lang), height=50)
with tts_col:
    st.session_state.qa_tts_enabled = st.toggle(
        t("qa_tts_toggle"), value=st.session_state.qa_tts_enabled
    )
if st.session_state.qa_tts_enabled:
    components.html(_stop_speaking_html(t("qa_stop_speaking")), height=46)

query = st.chat_input(t("qa_chat_placeholder")) or clicked

if query:
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

for query, answer, results in reversed(st.session_state.qa_history):
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander(t("qa_sources", n=len(results))):
            for score, doc in results:
                st.markdown(t("qa_source_relevance", title=doc["title"], score=f"{score:.2f}"))
                st.caption(doc["content"].strip()[:300] + "...")

# TTS playback — inject a 0-height iframe that speaks the latest answer
if st.session_state.qa_tts_enabled and st.session_state.qa_answer_tts:
    components.html(
        _tts_html(st.session_state.qa_answer_tts, st.session_state.qa_answer_counter, lang),
        height=0,
    )

if not st.session_state.qa_history:
    st.markdown("---")
    with st.expander(t("qa_kb_expander")):
        for doc in kb._docs:
            st.markdown(f"- **{doc['title']}**")
