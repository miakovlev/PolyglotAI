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

# Local ASR
try:
    from faster_whisper import WhisperModel
    _HAS_LOCAL = True
except Exception:
    _HAS_LOCAL = False

DATA_DIR = read_env("DATA_DIR", "app/data")
ensure_dir(DATA_DIR)


def _audio_duration_seconds(path: str) -> float:
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000.0
    except Exception:
        try:
            with contextlib.closing(wave.open(path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0


def transcribe(audio_path: str, engine: str = "openai", language: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns: { text, segments: [{start, end, text}], language }
    """
    engine = (engine or "openai").lower()

    if engine == "local":
        if not _HAS_LOCAL:
            raise RuntimeError("Local ASR requested but faster-whisper is not installed.")

        model_size = read_env("ASR_LOCAL_MODEL", "base")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments_iter, info = model.transcribe(audio_path, language=language, vad_filter=True)

        segs = []
        txts = []
        for s in segments_iter:
            segs.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            })
            txts.append(s.text.strip())

        return {
            "text": " ".join(txts).strip(),
            "segments": segs,
            "language": getattr(info, "language", language) or language
        }

    else:
        client = OpenAI(api_key=read_env("OPENAI_API_KEY"))
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language
            )

        # OpenAI Whisper returns 'text'
        text = getattr(resp, "text", "")

        # We may not get timestamps from API; keep empty list
        return {
            "text": text,
            "segments": [],
            "language": language
        }
