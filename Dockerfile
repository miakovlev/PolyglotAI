# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

ARG APP_USER=app
RUN useradd -m -u 10001 ${APP_USER}

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /app/.streamlit && chown -R ${APP_USER}:${APP_USER} /app
USER ${APP_USER}

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
