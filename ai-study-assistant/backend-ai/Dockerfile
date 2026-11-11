# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — builder: cài toàn bộ Python deps vào virtualenv /opt/venv
# (build-essential chỉ nằm ở stage này, không lọt vào image cuối)
# =============================================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv thay cho pip: resolve + cài nhanh hơn nhiều, và không bị "backtracking"
# kéo dài như pip (log cũ: pip tải lần lượt hàng chục bản langsmith, ~25s/bản).
RUN pip install --no-cache-dir "uv>=0.5,<1" && uv venv --seed /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_LINK_MODE=copy

# --- torch ------------------------------------------------------------------
# Mặc định cài torch bản CPU (~200MB) thay vì bản CUDA từ PyPI (kéo theo
# ~2-3GB gói nvidia-*). docker-compose.yml hiện không cấp GPU cho container
# nên bản CUDA không có tác dụng. Nếu sau này container cần GPU NVIDIA:
#   docker build --build-arg TORCH_INDEX_URL= ./backend-ai
# (để trống = cài torch mặc định từ PyPI).
# Đặt TRƯỚC `COPY requirements.txt` để đổi requirements.txt không làm tải lại torch.
ARG TORCH_VERSION=2.4.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        uv pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"; \
    fi

# --- phần còn lại của dependencies -------------------------------------------
WORKDIR /build
COPY requirements.txt .

# Chốt an toàn: TORCH_VERSION ở trên phải khớp requirements.txt, nếu không
# bước dưới sẽ âm thầm cài lại torch CUDA từ PyPI.
RUN grep -qE "^torch==${TORCH_VERSION}[[:space:]]*$" requirements.txt \
    || { echo "ERROR: TORCH_VERSION=${TORCH_VERSION} không khớp dòng torch== trong requirements.txt" >&2; exit 1; }

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Chỉ in thông tin để dễ kiểm tra trong log build (không làm fail build).
RUN uv pip list 2>/dev/null | grep -i -E '^(torch|nvidia)' || true

# Stage 2 — runtime: image cuối, chỉ có tesseract + venv đã build sẵn + code
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
