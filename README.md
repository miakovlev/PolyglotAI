# PolyglotAI · Language Toolkit (Streamlit)

🎧 A streamlined Streamlit app for language work: **audio transcription**, **text structuring**, **translation**, **text-to-speech (TTS)**, plus **phrase explanation** and **phrase translation** — all in one UI.

---

## ✨ Features

- **Upload & Transcribe** — upload `.mp3/.m4a/.wav`, transcribe via OpenAI (default `gpt-4o-mini-transcribe`).
- **Structure** — turn raw transcript into **dialogues** or **topic sections**.
- **Translate** — quick text translation.
- **TTS** — synthesize MP3 via OpenAI TTS (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`).
- **Explain phrase** — learner-friendly explanation in the **target** language.
- **Translate phrase** — translation pack with examples and notes.

Each browser visitor gets an isolated Streamlit session (`st.session_state`). App-level configs and secrets come from `secrets.toml`.

---

## 🧱 Project Structure

```
app/
  ui/
    main.py          # Streamlit UI: tabs, inputs, session state
  services/
    asr.py           # transcription via OpenAI
    llm.py           # chat/LLM helpers (structure/translate/explain)
    tts.py           # TTS to MP3 (stream to file)
    utils.py         # helpers: sha1, JSON IO, limits, etc.
```
---

## 🔑 Secrets & Config (TOML)

Configuration lives in `secrets.toml`.

### Local setup

Create `./.streamlit/secrets.toml`:

```toml
# ==== Required ====
OPENAI_API_KEY = "sk-your-key"
APP_TOKEN = "admin"         # optional gate; if set, the app asks for it in the sidebar

# ==== Models ====
MODEL = "gpt-5-mini"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"
ASR_MODEL = "gpt-4o-mini-transcribe"

# ==== Limits ====
MAX_AUDIO_MINUTES = 60

# ==== Paths ====
DATA_DIR = "app/data"
PYTHONPATH = "."
```

> App modules read the OpenAI key from `st.secrets["OPENAI_API_KEY"]`. (Single app-level key.)

---

## 🧩 Requirements

- **Python** 3.12 (Docker image uses `python:3.12-slim`)
- **ffmpeg** (required by `pydub` to decode audio)
- Python libs from `requirements.txt` (e.g., `streamlit`, `openai`, `pydub`, ...)

Install ffmpeg locally:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Windows: install official ffmpeg build and add it to `PATH`

---

## ▶️ Local Run (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# create ./.streamlit/secrets.toml as shown above

streamlit run app/ui/main.py
# open http://localhost:8501
```

---

## 🐳 Run with Docker / Docker Compose


### Build & Run

```bash
docker compose up --build
# or in background:
docker compose up -d --build
```

Open http://localhost:8501

---

## 📦 Data Layout

- `app/data/audio/` — uploaded audio files (name = `sha1` of content)
- `app/data/transcripts/` — JSON transcripts
- `app/data/tts/` — generated MP3 files

`DATA_DIR` is configurable in `secrets.toml` (default `app/data`).

---

## 🧠 Multi-User Notes

- Each user session is isolated via `st.session_state`.
- App-level secrets (`st.secrets`) are shared across users.
- Avoid concurrent writes to the same file on heavy traffic; consider a DB for shared state.