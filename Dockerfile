# ---- Build stage ---------------------------------------------------------
FROM python:3.11-slim AS base

# Metadata for Hugging Face Spaces / OpenEnv discovery
LABEL maintainer="content-moderation-env"
LABEL openenv="true"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Start the FastAPI server that exposes OpenEnv endpoints (/reset, /step)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
