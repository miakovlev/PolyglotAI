from pathlib import Path
import sys
from typing import Optional
import os
import pathlib

import streamlit as st

# make repo root importable (implicit namespace packages)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64
import hashlib
import hmac
import secrets
from typing import List

import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
import google.auth.transport.requests

# Use full scopes to avoid "scope changed" warnings
SCOPES: List[str] = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")

def _make_state(secret: str) -> str:
    # random nonce + HMAC(secret, nonce)
    nonce = secrets.token_urlsafe(16)
    sig = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).digest()
    return f"{nonce}.{_b64url(sig)}"

def _verify_state(state: str, secret: str) -> bool:
    try:
        nonce, sig_b64 = state.split(".", 1)
        expected = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).digest()
        return hmac.compare_digest(sig_b64, _b64url(expected))
    except Exception:
        return False

def _qp(params, key: str) -> str:
    v = params.get(key, "")
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""

def require_google_auth() -> str:
    """Authenticate via Google OAuth and return the user's email."""
    client_id = st.secrets.get("GOOGLE_CLIENT_ID")
    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = st.secrets.get("GOOGLE_REDIRECT_URI")
    allowed = st.secrets.get("ALLOWED_EMAILS", [])

    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    if not client_id or not client_secret:
        st.error("Google OAuth is not configured.")
        st.stop()

    params = st.query_params
    code = _qp(params, "code")
    state = _qp(params, "state")

    # 1) Start auth
    if not code:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=SCOPES,
            redirect_uri=redirect_uri,
        )
        signed_state = _make_state(client_secret)
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
            state=signed_state,
        )
        button_html = f"""
        <a href="{auth_url}" target="_self" style="text-decoration:none;">
            <div style="
                display:flex;
                align-items:center;
                justify-content:center;
                background-color:white;
                color:#444;
                border:1px solid #dadce0;
                border-radius:4px;
                padding:0.5em 1em;
                font-size:16px;
                font-weight:500;
                cursor:pointer;">
                <img src="https://developers.google.com/identity/images/g-logo.png" style="height:18px;margin-right:8px;">
                <span>Sign in with Google</span>
            </div>
        </a>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        st.stop()

    # 2) Callback: verify state
    if not _verify_state(state, client_secret):
        st.error("State mismatch.")
        st.stop()

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)

    request = google.auth.transport.requests.Request()
    info = id_token.verify_oauth2_token(flow.credentials.id_token, request, client_id)
    email = info.get("email")

    if allowed and email not in allowed:
        st.warning("Unauthorized email.")
        st.stop()

    st.session_state["user_email"] = email
    st.query_params.clear()
    return email


def logout() -> None:
    """Clear stored user session and rerun the app."""
    st.session_state.pop("user_email", None)
    st.rerun()

from app.services.utils import (
    ensure_dir,
    save_json,
    sha1_of_bytes,
    sha1_of_text,
    minutes_limit_ok,
)
from app.services.asr import transcribe, _audio_duration_seconds
from app.services.llm import structure_text, translate_text, explain_phrase, translate_phrase
from app.services.tts import tts_to_mp3

APP_TITLE = "PolyglotAI | Language Toolkit"
st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")
st.title("🎧PolyglotAI | Language Toolkit")

LANG_PRESETS = {
    "English": "en",
    "German": "de",
    "Greek": "el",
    "Russian": "ru",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Ukrainian": "uk",
}

def pick_language(
    label: str,
    key_prefix: str,
    default_name: Optional[str] = None,
    include_auto: bool = False,
) -> str:
    options = list(LANG_PRESETS.keys()) + ["Other…"]
    if include_auto:
        options = ["Auto-detect"] + options
    index = options.index(default_name) if (default_name in options) else 0
    choice = st.selectbox(label, options, index=index, key=f"{key_prefix}_preset")
    if choice == "Auto-detect":
        return ""
    if choice == "Other…":
        custom = st.text_input("Enter language (name or ISO code)", key=f"{key_prefix}_custom")
        return (custom or "").strip()
    return LANG_PRESETS[choice]

# ---- Google Login (moved to services/auth/google_oauth.py)
user_email = require_google_auth()

# ---- Sidebar
st.sidebar.header("Settings")
DATA_DIR = st.secrets.get("DATA_DIR", "app/data")
ensure_dir(DATA_DIR)

model_options = [
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]
default_model = st.secrets.get("MODEL", "gpt-5-mini")
model = st.sidebar.selectbox(
    "LLM model",
    model_options,
    index=model_options.index(default_model) if default_model in model_options else 0,
)

max_audio_minutes = int(st.secrets.get("MAX_AUDIO_MINUTES", "60"))
st.sidebar.caption(f"Signed in as {user_email}")
if st.sidebar.button("Log out"):
    logout()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["1) Upload & Transcribe", "2) Structure", "3) Translate", "4) TTS", "5) Explain phrase", "6) Translate phrase"]
)

if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "structured" not in st.session_state:
    st.session_state.structured = None
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "explanation_meta" not in st.session_state:
    st.session_state.explanation_meta = ""
if "tphrase" not in st.session_state:
    st.session_state.tphrase = ""
if "tphrase_meta" not in st.session_state:
    st.session_state.tphrase_meta = ""

# ====== 1) Upload & Transcribe ======
with tab1:
    st.subheader("Upload audio & transcribe")
    uploaded = st.file_uploader(
        "Audio file (.mp3/.m4a/.wav)", type=["mp3", "m4a", "wav"], accept_multiple_files=False
    )

    default_asr_model = st.secrets.get("ASR_MODEL", "gpt-4o-mini-transcribe")
    transcribe_model = st.selectbox("Transcription model", [default_asr_model], index=0)
    preferred_lang = st.text_input("Preferred language (optional)", "", placeholder="en, de, el, ru (blank = auto)")

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
            st.error(f"The file exceeds the {max_audio_minutes}-minute limit.")
        else:
            if st.button("Transcribe"):
                with st.spinner("Transcribing..."):
                    res = transcribe(audio_path, model=transcribe_model, language=(preferred_lang or None))
                st.session_state.transcript = res
                out_dir = os.path.join(DATA_DIR, "transcripts")
                ensure_dir(out_dir)
                save_json(os.path.join(out_dir, f"{sha}.json"), res)
                st.success("Transcription complete.")
                st.text_area("Transcript", res.get("text", ""), height=200)

# ====== 2) Structure ======
with tab2:
    st.subheader("Structure the transcript")
    mode_label = st.radio("Mode", ["Dialog", "Topics"], horizontal=True, index=0)
    mode_value = mode_label.lower()

    source_text = st.session_state.transcript.get("text", "") if st.session_state.transcript else ""
    source_text = st.text_area("Source text (used if no transcript above)", value=source_text, height=200)

    if st.button("Structure"):
        if not source_text.strip():
            st.warning("No text to structure.")
        else:
            with st.spinner("Structuring..."):
                out = structure_text(source_text, mode=mode_value, model=model)
            st.session_state.structured = out["structured_text"]
            st.text_area("Structured", st.session_state.structured, height=300)
            st.caption(f"Model: {out['model']} — Usage: {out['usage'].total_tokens} tokens")

# ====== 3) Translate ======
with tab3:
    st.subheader("Translate")
    text = st.text_area("Source text", height=160)
    tgt_lang = pick_language("Target language", key_prefix="translate_tgt", default_name="English")
    if st.button("Translate"):
        if not text.strip() or not tgt_lang.strip():
            st.warning("Please enter text and choose a target language.")
        else:
            with st.spinner("Translating..."):
                out = translate_text(text, tgt_lang, model=model)
            st.text_area("Translation", out["translation"], height=200)
            st.caption(f"Model: {out['model']} — Usage: {out['usage'].total_tokens} tokens")

# ====== 4) TTS ======
with tab4:
    st.subheader("Text-to-Speech")
    tts_text = st.text_area("Text to synthesize", height=160)
    tts_model_options = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
    default_tts_model = st.secrets.get("TTS_MODEL", "tts-1")
    tts_model = st.selectbox("TTS model", tts_model_options, index=tts_model_options.index(default_tts_model) if default_tts_model in tts_model_options else 0)
    voice_options = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]
    voice = st.selectbox("Voice", voice_options, index=0)
    if st.button("Generate MP3"):
        if not tts_text.strip():
            st.warning("Please enter some text.")
        else:
            out_dir = os.path.join(DATA_DIR, "tts")
            ensure_dir(out_dir)
            fname = sha1_of_text(tts_text)[:12] + ".mp3"
            out_path = os.path.join(out_dir, fname)
            with st.spinner("Synthesizing..."):
                res = tts_to_mp3(tts_text, out_path, model=tts_model, voice=voice)
            st.success(f"Saved: {res['path']}")
            st.caption(f"Model: {res['model']} — Voice: {res['voice']}")
            audio_bytes = open(out_path, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Download MP3", data=audio_bytes, file_name=fname, mime="audio/mpeg")

# ====== 5) Explain phrase ======
with tab5:
    st.subheader("Explain phrase")
    phr_col1, phr_col2, phr_col3 = st.columns([1, 1, 1])
    phrase = phr_col1.text_input("Phrase", placeholder="Type the phrase to explain...")
    src_lang = pick_language("Source language (optional)", key_prefix="explain_src", default_name="Auto-detect", include_auto=True)
    dst_lang = pick_language("Target language", key_prefix="explain_dst", default_name="German")
    if st.button("Explain"):
        if not phrase.strip() or not dst_lang.strip():
            st.warning("Provide phrase and target language. Source language is optional.")
        else:
            with st.spinner("Explaining..."):
                out = explain_phrase(phrase, source_lang=(src_lang or None), target_lang=dst_lang, model=model)
            st.session_state.explanation = out["explanation"]
            st.session_state.explanation_meta = f"Model: {out['model']} — Usage: {out['usage'].total_tokens} tokens"
    if st.session_state.explanation:
        st.markdown(st.session_state.explanation)
        if st.session_state.explanation_meta:
            st.caption(st.session_state.explanation_meta)

# ====== 6) Translate phrase ======
with tab6:
    st.subheader("Translate phrase")
    tr_col1, tr_col2, tr_col3 = st.columns([1, 1, 1])
    tr_phrase = tr_col1.text_input("Phrase to translate", placeholder="Enter a phrase...")
    tr_src = pick_language("Source language (optional)", key_prefix="tphrase_src", default_name="Auto-detect", include_auto=True)
    tr_dst = pick_language("Target language", key_prefix="tphrase_dst", default_name="English")
    if st.button("Translate phrase"):
        if not tr_phrase.strip() or not tr_dst.strip():
            st.warning("Provide a phrase and choose the target language. Source language is optional.")
        else:
            with st.spinner("Translating..."):
                out = translate_phrase(tr_phrase, target_lang=tr_dst, source_lang=(tr_src or None), model=model)
            st.session_state.tphrase = out["analysis"]
            st.session_state.tphrase_meta = f"Model: {out['model']} — Usage: {out['usage'].total_tokens} tokens"
    if st.session_state.tphrase:
        st.markdown(st.session_state.tphrase)
        if st.session_state.tphrase_meta:
            st.caption(st.session_state.tphrase_meta)

st.divider()
st.caption("Local MVP. Everyone uses their own API key. Built with Streamlit, OpenAI.")