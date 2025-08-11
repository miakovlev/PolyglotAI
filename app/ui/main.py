import os
import io
import time
import pathlib
import base64

import streamlit as st
from dotenv import load_dotenv

from app.services.utils import (
    read_env,
    ensure_dir,
    save_json,
    load_json,
    sha1_of_bytes,
    sha1_of_text,
    minutes_limit_ok,
)
from app.services.asr import transcribe, _audio_duration_seconds
from app.services.llm import structure_text, translate_text, explain_phrase, chat
from app.services.rag import InMemoryIndex, simple_chunk
from app.services.tts import tts_to_mp3

# Load .env early
load_dotenv(override=True)

APP_TITLE = "DS Language Toolkit — MVP"
st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")
st.title("🎧 DS Language Toolkit — MVP")

# --- Auth (simple) ---
APP_TOKEN = read_env("APP_TOKEN", "")
if APP_TOKEN:
    token = st.sidebar.text_input("Enter APP_TOKEN", type="password")
    if token != APP_TOKEN:
        st.warning("Enter APP_TOKEN to unlock the app.")
        st.stop()

# --- Sidebar settings ---
st.sidebar.header("Settings")
DATA_DIR = read_env("DATA_DIR", "app/data")
ensure_dir(DATA_DIR)

model = st.sidebar.text_input("LLM model", read_env("MODEL", "gpt-4o-mini"))
embed_model = st.sidebar.text_input(
    "Embedding model", read_env("EMBED_MODEL", "text-embedding-3-small")
)
asr_engine = st.sidebar.selectbox(
    "ASR engine",
    ["openai", "local"],
    index=0 if read_env("ASR_ENGINE", "openai") == "openai" else 1,
)
asr_local_model = st.sidebar.text_input(
    "Local Whisper model", read_env("ASR_LOCAL_MODEL", "base")
)
max_audio_minutes = int(read_env("MAX_AUDIO_MINUTES", "60"))
top_k = int(read_env("TOP_K", "5"))
max_answer_tokens = int(read_env("MAX_ANSWER_TOKENS", "800"))

st.sidebar.caption("Keys are read from your local .env. Tokens are billed to YOUR account.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1) Upload & Transcribe",
        "2) Structure",
        "3) Ask (QA)",
        "4) Translate",
        "5) TTS",
        "6) Explain phrase",
    ]
)

# --- Session state ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "structured" not in st.session_state:
    st.session_state.structured = None
if "index" not in st.session_state:
    st.session_state.index = None

# ====== 1) Upload & Transcribe ======
with tab1:
    st.subheader("Upload audio and transcribe")
    uploaded = st.file_uploader(
        "Audio file (.mp3/.m4a/.wav)", type=["mp3", "m4a", "wav"], accept_multiple_files=False
        # Note: Streamlit handles temporary storage; we persist below.
    )
    colA, colB = st.columns([1, 1])
    with colA:
        preferred_lang = st.text_input("Preferred language (optional, e.g., en, de, ru)", "")
    with colB:
        st.write("")
        st.write(f"Engine: {asr_engine}")

    if uploaded is not None:
        audio_bytes = uploaded.read()
        sha = sha1_of_bytes(audio_bytes)
        audio_dir = os.path.join(DATA_DIR, "audio")
        ensure_dir(audio_dir)
        ext = pathlib.Path(uploaded.name).suffix or ".mp3"
        audio_path = os.path.join(audio_dir, f"{sha}{ext}")
        if not os.path.exists(audio_path):
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

        dur = _audio_duration_seconds(audio_path)
        st.info(f"Saved as: `{audio_path}` — duration ~ {dur/60.0:.1f} min")

        if not minutes_limit_ok(dur, max_audio_minutes):
            st.error(f"File longer than {max_audio_minutes} minutes limit.")
        else:
            if st.button("Transcribe"):
                with st.spinner("Transcribing..."):
                    res = transcribe(
                        audio_path, engine=asr_engine, language=(preferred_lang or None)
                    )
                st.session_state.transcript = res
                out_dir = os.path.join(DATA_DIR, "transcripts")
                ensure_dir(out_dir)
                save_json(os.path.join(out_dir, f"{sha}.json"), res)
                st.success("Transcription done.")
                st.text_area("Transcript", res.get("text", ""), height=200)

# ====== 2) Structure ======
with tab2:
    st.subheader("Structure the transcript")
    mode = st.radio("Mode", ["dialog", "topics"], horizontal=True, index=0)

    source_text = ""
    if st.session_state.transcript:
        source_text = st.session_state.transcript.get("text", "")
    source_text = st.text_area("Source text (fallback if no transcript)", value=source_text, height=200)

    if st.button("Format"):
        if not source_text.strip():
            st.warning("No text to structure.")
        else:
            with st.spinner("Formatting..."):
                out = structure_text(source_text, mode=mode)
            st.session_state.structured = out["structured_text"]
            st.text_area("Structured", st.session_state.structured, height=300)
            st.caption(f"Usage: {out['usage'].total_tokens} tokens")

# ====== 3) Ask (QA) ======
with tab3:
    st.subheader("Ask questions about the text")
    base_text = st.session_state.structured or (
        st.session_state.transcript.get("text", "") if st.session_state.transcript else ""
    )
    base_text = st.text_area("Text to index", value=base_text, height=200)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Build index / Rebuild"):
            if not base_text.strip():
                st.warning("No text.")
            else:
                segments = [{"text": t} for t in simple_chunk(base_text)]
                idx = InMemoryIndex(embed_model=embed_model)
                with st.spinner("Embedding & indexing..."):
                    idx.build(segments)
                st.session_state.index = idx
                st.success(f"Built index with {len(segments)} chunks.")

    with col2:
        question = st.text_input("Your question")
        ask = st.button("Ask")
        if ask:
            if not st.session_state.index:
                st.warning("Build the index first.")
            else:
                hits = st.session_state.index.query(question, top_k=top_k)
                ctx = []
                for i, h in enumerate(hits, 1):
                    ctx.append(f"[{i}] {h['text']}")
                ctx_text = "\n\n".join(ctx) if ctx else "(no context found)"

                system = (
                    "Answer strictly based on the provided context. "
                    "If not present, say you don't know. Cite as [1],[2],..."
                )
                user = f"Question:\n{question}\n\nContext:\n{ctx_text}"

                with st.spinner("Thinking..."):
                    ans = chat(
                        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        model=model,
                    )
                st.markdown(ans["content"])
                st.caption(f"Usage: {ans['usage'].total_tokens} tokens")

# ====== 4) Translate ======
with tab4:
    st.subheader("Translate")
    text = st.text_area("Text", height=160)
    tgt = st.text_input("Target language (e.g., en, de, ru)")
    if st.button("Translate"):
        if not text.strip() or not tgt.strip():
            st.warning("Provide text and target language.")
        else:
            with st.spinner("Translating..."):
                out = translate_text(text, tgt)
            st.text_area("Translation", out["translation"], height=200)
            st.caption(f"Usage: {out['usage'].total_tokens} tokens")

# ====== 5) TTS ======
with tab5:
    st.subheader("Text-to-Speech (4o TTS)")
    tts_text = st.text_area("Text to voice", height=160)
    voice = st.text_input("Voice", read_env("TTS_VOICE", "alloy"))
    if st.button("Generate MP3"):
        if not tts_text.strip():
            st.warning("Provide text.")
        else:
            out_dir = os.path.join(DATA_DIR, "tts")
            ensure_dir(out_dir)
            fname = sha1_of_text(tts_text)[:12] + ".mp3"
            out_path = os.path.join(out_dir, fname)
            with st.spinner("Synthesizing..."):
                res = tts_to_mp3(tts_text, out_path)
            st.success(f"Saved: {res['path']}")
            audio_bytes = open(out_path, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Download MP3", data=audio_bytes, file_name=fname, mime="audio/mpeg")

# ====== 6) Explain phrase ======
with tab6:
    st.subheader("Explain phrase")
    phr_col1, phr_col2, phr_col3 = st.columns([1, 1, 1])
    with phr_col1:
        phrase = st.text_input("Phrase")
    with phr_col2:
        src = st.text_input("Source lang (e.g., ru)")
    with phr_col3:
        dst = st.text_input("Target lang (e.g., de)")

    if st.button("Explain"):
        if not phrase.strip() or not src.strip() or not dst.strip():
            st.warning("Provide phrase, source and target languages.")
        else:
            out = explain_phrase(phrase, src, dst)
            st.markdown(out["explanation"])
            st.caption(f"Usage: {out['usage'].total_tokens} tokens")

st.divider()
st.caption("Local MVP. Everyone uses their own API key. Built with Streamlit, FAISS, Whisper, OpenAI.")
