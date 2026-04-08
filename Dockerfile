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

# Override default command to satisfy Hugging Face Spaces health check 
# (HF expects a long-running process binding to port 7860).
# The OpenEnv evaluator will still use `inference.py` via `openenv.yaml`.
CMD ["python", "-m", "http.server", "7860"]
