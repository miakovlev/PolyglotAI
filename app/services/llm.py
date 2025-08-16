from typing import List, Dict, Any, Optional
from openai import OpenAI
from .utils import read_env, parse_openai_usage, Usage


def chat(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
    client = OpenAI(api_key=read_env("OPENAI_API_KEY"))
    m = model or read_env("MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=m,
        messages=messages,
        temperature=temperature
    )
    content = resp.choices[0].message.content
    usage = parse_openai_usage(resp)
    return {"content": content, "usage": usage}


def structure_text(raw_text: str, mode: str = "dialog") -> Dict[str, Any]:
    """
    mode: 'dialog' or 'topics'
    Returns: { structured_text: str, json_hint: Optional[str], usage }
    """
    system = "You reformulate transcripts into clean study-ready text."

    if mode == "dialog":
        user = f"""Carefully analyze the following transcript and identify distinct, separate dialogues.
For each distinct dialogue, format it as a clean conversation with speaker turns.
If the transcript contains multiple distinct dialogues, separate them clearly with a noticeable separator, like a line of dashes.

For each dialogue:
- Merge broken lines and fix obvious punctuation.
- Keep natural short paragraphs.
- Assign speaker roles: If speakers aren't labeled, use 'Speaker A', 'Speaker B', 'Speaker C', etc., ensuring these labels are consistent and sequential across ALL identified dialogues in the entire transcript.

Transcript:
{raw_text}
"""
    else:
        user = f"""Split the transcript into topics/sections with headings.
If the transcript contains text from several distinct videos or segments, treat each as a separate major section.

Provide a concise outline and the cleaned text per section.

Fix punctuation lightly, do not invent facts.

Transcript:
{raw_text}
"""

    out = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return {"structured_text": out["content"], "usage": out["usage"]}


def translate_text(text: str, target_lang: str) -> Dict[str, Any]:
    system = "You are a precise translator. Preserve meaning and tone. Return only the translation."
    out = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Translate to {target_lang}:\n{text}"}
        ]
    )
    return {"translation": out["content"], "usage": out["usage"]}


def explain_phrase(phrase: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
    system = (
        "You are a language tutor. Provide translation, grammar notes, "
        "common usages, 3 example sentences, and 3 collocations."
    )
    user = (
        f"Phrase: {phrase}\n"
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        "Format as markdown with clear bullet points."
    )
    out = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return {"explanation": out["content"], "usage": out["usage"]}
