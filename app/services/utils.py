import os
import json
import hashlib
import datetime
import pathlib
from dataclasses import dataclass
from typing import Optional


def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_of_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def parse_openai_usage(resp) -> Usage:
    try:
        u = resp.usage
        return Usage(
            getattr(u, "prompt_tokens", 0),
            getattr(u, "completion_tokens", 0),
            getattr(u, "total_tokens", 0),
        )
    except Exception:
        return Usage()


def minutes_limit_ok(file_seconds: float, max_minutes: int) -> bool:
    return (file_seconds / 60.0) <= max_minutes