"""ONNX Runtime (INT8) thay thế cho SentenceTransformer / CrossEncoder.

Hai lớp ``OnnxEmbedder`` và ``OnnxCrossEncoder`` có đúng giao diện mà ``vector_store.py`` đang gọi:

* ``OnnxEmbedder.encode(texts, show_progress_bar=False)`` — giống ``SentenceTransformer.encode``:
  nhận ``str`` -> mảng 1-D, nhận ``list[str]`` -> mảng 2-D. Mean pooling, KHÔNG chuẩn hóa
  (đúng như paraphrase-multilingual-mpnet-base-v2; Qdrant tự chuẩn hóa khi dùng cosine).
* ``OnnxCrossEncoder.predict(pairs)`` — giống ``CrossEncoder.predict``: model 1 nhãn => sigmoid.

Thư mục model do ``scripts/quantize_models.py`` tạo:

    <rag_onnx_dir>/<tên-model-an-toàn>/
        model.int8.onnx   (và/hoặc model.fp32.onnx)
        tokenizer.json, ... (file tokenizer, để chạy offline)
        meta.json         (loại model, max_length, kết quả so khớp với bản gốc)

Không import torch ở đây — chạy được trong image chỉ có onnxruntime + tokenizers.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

META_FILE = "meta.json"


def safe_model_dirname(model_name: str) -> str:
    """'org/name' -> 'org__name' (dùng chung cho script xuất model và lúc chạy)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name.strip())


def model_file(model_dir: Union[str, Path], precision: str) -> Path:
    return Path(model_dir) / f"model.{precision}.onnx"


def find_model_dir(root: Union[str, Path], model_name: str, precision: str) -> Optional[Path]:
    """Trả về thư mục model nếu đủ file cần dùng, ngược lại None (=> caller dùng PyTorch)."""
    directory = Path(root) / safe_model_dirname(model_name)
    if model_file(directory, precision).is_file() and (directory / META_FILE).is_file():
        return directory
    return None


def read_meta(model_dir: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(model_dir) / META_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def _make_session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads and threads > 0:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    # Tokenizer nằm sẵn trong thư mục model => không cần mạng / HF hub lúc chạy.
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


class _OnnxBase:
    def __init__(self, model_dir: Union[str, Path], precision: str = "int8", threads: int = 0):
        self.model_dir = Path(model_dir)
        self.precision = precision
        self.meta = read_meta(self.model_dir)
        self.max_length = int(self.meta.get("max_length", 128))
        self.tokenizer = _load_tokenizer(self.model_dir)
        # Tokenizer "fast" của HuggingFace đổi trạng thái truncation/padding theo từng lời gọi nên dùng
        # chung giữa nhiều thread (FastAPI threadpool) có thể gây "RuntimeError: Already borrowed".
        # Tokenize rất nhanh nên khóa ở đây; phần nặng (ONNX Runtime, thread-safe) chạy ngoài khóa.
        self._tok_lock = threading.Lock()
        self.session = _make_session(model_file(self.model_dir, precision), threads)
        self._input_names = [i.name for i in self.session.get_inputs()]

    def _feed(self, enc: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feed = {}
        for name in self._input_names:
            if name in enc:
                feed[name] = enc[name].astype(np.int64, copy=False)
            elif name == "token_type_ids":  # model có input này nhưng tokenizer không sinh ra
                feed[name] = np.zeros_like(enc["input_ids"], dtype=np.int64)
            else:
                raise KeyError(f"ONNX model needs input '{name}' that the tokenizer did not provide")
        return feed


class OnnxEmbedder(_OnnxBase):
    """Tương đương ``SentenceTransformer(...).encode`` cho model Transformer + mean pooling."""

    def __init__(self, model_dir, precision: str = "int8", threads: int = 0, batch_size: int = 32):
        super().__init__(model_dir, precision, threads)
        self.batch_size = batch_size

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        with self._tok_lock:
            enc = self.tokenizer(
                texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="np"
            )
        hidden = self.session.run(None, self._feed(enc))[0]  # (batch, seq, dim)
        mask = enc["attention_mask"].astype(np.float32)[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return (summed / counts).astype(np.float32)

    def encode(self, sentences: Union[str, Sequence[str]], show_progress_bar: bool = False,
               batch_size: Optional[int] = None, **_: Any) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        step = batch_size or self.batch_size
        # Sắp theo độ dài để mỗi batch ít padding (nhanh hơn rõ rệt trên CPU), rồi trả lại đúng thứ tự.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: List[Optional[np.ndarray]] = [None] * len(texts)
        for start in range(0, len(order), step):
            idx = order[start:start + step]
            vectors = self._encode_batch([texts[i] for i in idx])
            for i, vec in zip(idx, vectors):
                out[i] = vec
        result = np.stack(out)  # type: ignore[arg-type]
        return result[0] if single else result


class OnnxCrossEncoder(_OnnxBase):
    """Tương đương ``CrossEncoder(...).predict`` (model 1 nhãn => sigmoid, như sentence-transformers)."""

    def __init__(self, model_dir, precision: str = "int8", threads: int = 0, batch_size: int = 16):
        super().__init__(model_dir, precision, threads)
        self.batch_size = batch_size

    def predict(self, sentences: Sequence[Tuple[str, str]], **_: Any) -> np.ndarray:
        pairs = list(sentences)
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores = np.zeros(len(pairs), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            with self._tok_lock:
                enc = self.tokenizer(
                    [pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                    padding=True, truncation="longest_first", max_length=self.max_length,
                    return_tensors="np",
                )
            logits = self.session.run(None, self._feed(enc))[0]
            if logits.ndim == 2 and logits.shape[1] == 1:
                values = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            elif logits.ndim == 2:  # nhiều nhãn: lấy nhãn "relevant" (cột cuối)
                values = logits[:, -1]
            else:
                values = logits.reshape(len(idx))
            scores[idx] = values
        return scores


def default_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # Windows / macOS
        return max(1, os.cpu_count() or 1)
