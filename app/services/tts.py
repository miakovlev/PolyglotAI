import os
from typing import Dict, Any
from .utils import read_env, ensure_dir
from openai import OpenAI


def tts_to_mp3(text: str, out_path: str) -> Dict[str, Any]:
    client = OpenAI(api_key=read_env("OPENAI_API_KEY"))
    model = read_env("TTS_MODEL", "gpt-4o-mini-tts")
    voice = read_env("TTS_VOICE", "alloy")

    ensure_dir(os.path.dirname(out_path))

    # OpenAI Audio API: text-to-speech
    # Some SDK versions use audio.speech.create(), others audio.speech.with_streaming_response
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        format="mp3"
    )

    # Save bytes
    with open(out_path, "wb") as f:
        f.write(resp.read())

    return {"path": out_path}
