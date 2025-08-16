from typing import List, Dict, Any, Optional
from openai import OpenAI
from .utils import read_env, parse_openai_usage, Usage


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    client = OpenAI(api_key=read_env("OPENAI_API_KEY"))
    m = model or read_env("MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=m,
        messages=messages,
    )
    content = resp.choices[0].message.content
    usage = parse_openai_usage(resp)
    return {"content": content, "usage": usage, "model": m}


def structure_text(
    raw_text: str,
    mode: str = "dialog",
    model: Optional[str] = None,
) -> Dict[str, Any]:
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
            {"role": "user", "content": user},
        ],
        model=model,
    )
    return {"structured_text": out["content"], "usage": out["usage"], "model": out["model"]}


def translate_text(
    text: str,
    target_lang: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    system = (
        "You are a precise translator. Preserve meaning and tone. Return only the translation."
    )
    out = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Translate to {target_lang}:\n{text}"},
        ],
        model=model,
    )
    return {"translation": out["content"], "usage": out["usage"], "model": out["model"]}


def explain_phrase(
    phrase: str,
    source_lang: Optional[str],
    target_lang: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explain a phrase for a learner.
    - `phrase`: the original phrase in the source language
    - `source_lang`: language of the phrase (optional, can be auto-detected)
    - `target_lang`: language in which the explanation should be written

    The entire explanation (headings, commentary, grammar notes, etc.)
    MUST be written in the target language.

    Sections:
      1) Translation into the target language (1–2 best options)
      2) Literal meaning (if different)
      3) Grammar/morphology notes
      4) Usage & nuance
      5) 3 example sentences (source + target translation)
      6) 3 collocations
    """
    system = (
        "You are a language tutor. If source language is not provided, auto-detect it.\n"
        "ALL OUTPUT (headings and narrative text) MUST BE IN THE TARGET LANGUAGE requested by the user.\n"
        "Provide a clear, concise learner-friendly explanation with sections:\n"
        "1) Translation into the target language (1–2 best options)\n"
        "2) Literal meaning (if different)\n"
        "3) Grammar/Morphology notes (tense, aspect, case, gender, word class, etc.)\n"
        "4) Usage & nuance (register/tone; when appropriate/inappropriate)\n"
        "5) Examples: 3 sentences that illustrate the meaning; include the SOURCE sentence and the TARGET translation\n"
        "6) Collocations: 3 common or idiomatic combinations.\n"
        "Format as markdown with headings and bullet points. Be accurate and concise."
    )

    src_display = source_lang.strip() if (source_lang and source_lang.strip()) else "(auto-detect)"
    user = (
        f"Phrase (in source language): {phrase}\n"
        f"Source language: {src_display}\n"
        f"TARGET language for the entire explanation: {target_lang}\n"
        "Return the explanation in the TARGET language only."
    )

    out = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
    )
    return {"explanation": out["content"], "usage": out["usage"], "model": out["model"]}


def translate_phrase(
    phrase: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Translate a phrase into a target language and return a structured analysis pack.

    - `phrase`: the original phrase
    - `source_lang`: language of the phrase (optional, can be auto-detected)
    - `target_lang`: language into which translations/examples should be provided

    The ENTIRE analysis text (headings, commentary, notes) MUST be written in the SOURCE language.
    If source_lang is not specified, it should be auto-detected and used as the explanation language.

    Sections:
      1) Primary translations into the target language (1–3), each with usage notes
      2) Alternative phrasings/synonyms in the target language
      3) Literal translation and pitfalls/false friends
      4) Grammar snapshot of the source phrase
      5) 3–5 example sentences in the TARGET language, each followed by a gloss in the SOURCE language
      6) Optional: formal vs informal variants
    """
    system = (
        "You are an expert translator and language coach. If source language is not given, auto-detect it.\n"
        "ALL META TEXT (headings, explanations, notes) MUST BE IN THE SOURCE LANGUAGE.\n"
        "Deliver a compact but rich 'translation pack' for a single phrase with these sections:\n"
        "1) Primary translations into the target language (1–3) — each with a short note on usage/register/nuance.\n"
        "2) Alternative phrasings or synonyms in the target language (idiomatic or contextual), with brief explanations.\n"
        "3) Literal translation (if applicable) and warnings about false friends/pitfalls.\n"
        "4) Grammar snapshot in the SOURCE language: part of speech, key morphology/syntax.\n"
        "5) Examples (3–5): sentences in the TARGET language + a gloss/translation in the SOURCE language under each.\n"
        "6) (Optional) formal vs informal variants in the target language if relevant.\n"
        "Use markdown headings and bullet points. Do NOT output YAML/JSON. Be correct and concise."
    )

    src_display = source_lang.strip() if (source_lang and source_lang.strip()) else "(auto-detect)"
    user = (
        f"Phrase (in source language): {phrase}\n"
        f"Source language for analysis text: {src_display}\n"
        f"Target language for translations/examples: {target_lang}\n"
        "Return the entire analysis (headings + explanations) in the SOURCE language; "
        "translations and example sentences themselves are in the TARGET language, "
        "with a source-language gloss under each example."
    )

    out = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
    )
    return {"analysis": out["content"], "usage": out["usage"], "model": out["model"]}
