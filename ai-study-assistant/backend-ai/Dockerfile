# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — builder: install all Python dependencies into the /opt/venv virtual environment
# (build-essential is used only in this stage and is not included in the final image)
# =============================================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use uv instead of pip: dependency resolution and installation are much faster, avoiding pip's prolonged
# "backtracking" (previous logs showed pip downloading dozens of langsmith versions, ~25 seconds per version).
RUN pip install --no-cache-dir "uv>=0.5,<1" && uv venv --seed /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_LINK_MODE=copy

# --- torch ------------------------------------------------------------------
# By default, install the CPU version of torch (~200 MB) instead of the CUDA version from PyPI (which pulls in
# ~2-3 GB of nvidia-* packages). docker-compose.yml does not currently grant the container GPU access,
# so the CUDA build provides no benefit. If the container later needs an NVIDIA GPU:
#   docker build --build-arg TORCH_INDEX_URL= ./backend-ai
# (leave empty to install the default torch package from PyPI).
# Place this BEFORE `COPY requirements.txt` so changes to requirements.txt do not reinstall torch.
ARG TORCH_VERSION=2.4.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        uv pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"; \
    fi

# --- remaining dependencies ---------------------------------------------------
WORKDIR /build
COPY requirements.txt .

# Safety check: TORCH_VERSION above must match requirements.txt; otherwise,
# the step below may silently reinstall the CUDA build of torch from PyPI.
RUN grep -qE "^torch==${TORCH_VERSION}[[:space:]]*$" requirements.txt \
    || { echo "ERROR: TORCH_VERSION=${TORCH_VERSION} không khớp dòng torch== trong requirements.txt" >&2; exit 1; }

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Print information for easier inspection of build logs (does not fail the build).
RUN uv pip list 2>/dev/null | grep -i -E '^(torch|nvidia)' || true

# =============================================================================
# Stage 2 — runtime: final image containing only tesseract, the built virtual environment, and the application code
# =============================================================================
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# --- Run as an unprivileged user, NOT root ------------------------------------
# If the application is exploited (malicious upload, vulnerable dependency, etc.), an attacker has only the
# `app` user's permissions rather than root inside the container. ONNX models and the Hugging Face cache are the only writable
# directories; they are OUTSIDE /app (docker-compose mounts named volumes there). Do not place them under /app because /app
# is bind-mounted in development: a nested volume may be owned by root, preventing the `app` user from writing to it.
ARG APP_UID=10001
RUN groupadd --gid ${APP_UID} app \
    && useradd --uid ${APP_UID} --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /app /home/app/models /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    RAG_ONNX_DIR=/home/app/models/onnx

COPY --chown=app:app . .

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
