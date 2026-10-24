FROM python:3.11-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

FROM base as deps

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

FROM deps as runtime

# Fix outlines broken airport import (pyairports stub doesn't have airports module)
RUN sed -i "s/from pyairports.airports import AIRPORT_LIST/AIRPORT_LIST = []  # stub/" /usr/local/lib/python3.11/site-packages/outlines/types/airports.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
