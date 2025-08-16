import os
import io
import time
import tempfile
import wave
import contextlib
from typing import Dict, Any, List, Optional

from pydub import AudioSegment
from openai import OpenAI

from .utils import read_env, ensure_dir

DATA_DIR = read_env("DATA_DIR", "app/data")
ensure_dir(DATA_DIR)


def _audio_duration_seconds(path: str) -> float:
    """Utility: get audio duration in seconds."""
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000.0
    except Exception:
        try:
            with contextlib.closing(wave.open(path, "r")) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0


def transcribe(
    audio_path: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs,  # for backward-compatibility (e.g., engine=...)
) -> Dict[str, Any]:
    """
    Transcribe audio using OpenAI only.

    Args:
        audio_path: path to the audio file.
        model: OpenAI transcription model, e.g. "gpt-4o-mini-transcribe" (default from env ASR_MODEL or this value).
        language: optional BCP-47 language code (e.g., "en", "de", "ru").
        **kwargs: ignored (kept for compatibility if caller passes engine=...).

    Returns:
        {
          "text": str,
          "segments": [],          # timestamps are not returned by this endpoint
          "language": Optional[str]
        }
    """
    chosen_model = model or read_env("ASR_MODEL", "gpt-4o-mini-transcribe")
    client = OpenAI(api_key=read_env("OPENAI_API_KEY"))

    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=chosen_model,
            file=f,
            language=language
        )

    text = getattr(resp, "text", "") or ""
    return {
        "text": text.strip(),
        "segments": [],  # API не отдаёт таймкоды — оставляем пустым
        "language": language,
    }
