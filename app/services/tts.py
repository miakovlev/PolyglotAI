import os
from typing import Dict, Any, Optional

import streamlit as st
from openai import OpenAI

from .utils import ensure_dir


def tts_to_mp3(
    text: str,
    out_path: str,
    model: Optional[str] = None,
    voice: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize text into an MP3 file using OpenAI's streaming TTS API."""
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    m = model or st.secrets.get("TTS_MODEL", "gpt-4o-mini-tts")
    v = voice or st.secrets.get("TTS_VOICE", "alloy")

    ensure_dir(os.path.dirname(out_path))

    # Stream the audio directly to disk to avoid buffering large responses in memory
    with client.audio.speech.with_streaming_response.create(
        model=m,
        voice=v,
        input=text,
    ) as response:
        response.stream_to_file(out_path)

    return {"path": out_path, "model": m, "voice": v}

