"""Xuất model Embedding + Re-ranker sang ONNX và lượng tử hóa INT8, rồi ĐO độ lệch so với bản gốc.

Chạy MỘT LẦN (trong container backend, cần mạng để tải model từ HuggingFace):

    docker compose exec backend-ai python -m scripts.quantize_models

Kết quả nằm ở ``models/onnx/<tên-model>/`` (nằm trong volume ./backend-ai nên còn sau khi restart).
Backend tự dùng bản ONNX khi file tồn tại; xóa thư mục đó (hoặc đặt RAG_ONNX_ENABLED=false) để
quay về PyTorch.

Script KHÔNG chỉ "xuất file": nó so sánh bản INT8 với bản PyTorch gốc trên các câu mẫu tiếng Việt/Anh
và in ra:
  * embedding: cosine trung bình / thấp nhất giữa vector gốc và vector INT8
  * re-ranker: tương quan hạng (Spearman) và tỉ lệ trùng top-1
  * tốc độ: thời gian trung vị mỗi lần gọi, PyTorch vs ONNX INT8
Nếu chất lượng thấp hơn ngưỡng, script thoát với mã 2 và in cảnh báo — đừng dùng bản INT8 đó.

Lưu ý: đổi embedding sang INT8 làm vector lệch nhẹ so với vector đã lưu trong Qdrant (tính bằng fp32).
Truy vấn vẫn hoạt động, nhưng để chính xác nhất hãy upload lại tài liệu sau khi bật INT8.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from app.config import settings
from app.rag import onnx_models

SAMPLE_TEXTS = [
    "Lập trình hướng đối tượng có bốn tính chất cơ bản: đóng gói, kế thừa, đa hình và trừu tượng.",
    "Stack là cấu trúc dữ liệu hoạt động theo nguyên tắc vào sau ra trước (LIFO).",
    "Cơ sở dữ liệu quan hệ dùng khóa chính và khóa ngoại để liên kết các bảng.",
    "Kiến trúc máy tính gồm CPU, bộ nhớ, bus và các thiết bị vào/ra.",
    "A binary search tree keeps smaller keys on the left and larger keys on the right.",
    "HTTP là giao thức không trạng thái; cookie và session giúp máy chủ nhớ người dùng.",
    "Normalization reduces redundancy by splitting data into related tables.",
    "Thuật toán sắp xếp nhanh (quick sort) có độ phức tạp trung bình O(n log n).",
    "Tôi đã nộp bài tập lớn môn Phát triển ứng dụng web vào tuần trước.",
    "What is the difference between a process and a thread?",
]

RERANK_SETS: List[Tuple[str, List[str]]] = [
    (
        "Stack hoạt động theo nguyên tắc nào?",
        [
            "Stack là cấu trúc dữ liệu hoạt động theo nguyên tắc vào sau ra trước (LIFO).",
            "Hàng đợi (queue) hoạt động theo nguyên tắc vào trước ra trước (FIFO).",
            "Cây nhị phân tìm kiếm giữ các khóa nhỏ hơn ở bên trái.",
            "Khóa ngoại liên kết hai bảng trong cơ sở dữ liệu quan hệ.",
            "CPU thực hiện lệnh theo chu kỳ nạp, giải mã và thực thi.",
            "Cookie lưu thông tin phiên làm việc ở phía trình duyệt.",
        ],
    ),
    (
        "What is database normalization?",
        [
            "Normalization reduces redundancy by splitting data into related tables.",
            "Quick sort has an average complexity of O(n log n).",
            "Chuẩn hóa dữ liệu giúp giảm trùng lặp và tránh dị thường khi cập nhật.",
            "Stack là cấu trúc dữ liệu LIFO.",
            "HTTP là giao thức không trạng thái.",
            "Bộ nhớ đệm giúp CPU truy cập dữ liệu nhanh hơn.",
        ],
    ),
    (
        "Tính đa hình trong OOP là gì?",
        [
            "Đa hình cho phép cùng một lời gọi phương thức có hành vi khác nhau tùy kiểu đối tượng.",
            "Kế thừa cho phép lớp con dùng lại thuộc tính của lớp cha.",
            "Đóng gói che giấu dữ liệu bên trong đối tượng.",
            "Khóa chính xác định duy nhất mỗi dòng trong bảng.",
            "Session lưu trạng thái đăng nhập trên máy chủ.",
            "Bus địa chỉ xác định vị trí ô nhớ cần truy cập.",
        ],
    ),
]


# --------------------------------------------------------------------------------------
# Xuất ONNX + lượng tử hóa
# --------------------------------------------------------------------------------------
def _load_hf(name: str, kind: str):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    auto = AutoModel if kind == "embedding" else AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(name)
    try:  # attention "eager" xuất ONNX ổn định hơn SDPA
        model = auto.from_pretrained(name, attn_implementation="eager")
    except (TypeError, ValueError):
        model = auto.from_pretrained(name)
    return model.eval(), tokenizer


def _input_names(tokenizer) -> List[str]:
    return [n for n in tokenizer.model_input_names if n in ("input_ids", "attention_mask", "token_type_ids")]


def export_fp32(model, tokenizer, kind: str, target: Path, max_length: int, opset: int) -> None:
    import torch

    names = _input_names(tokenizer)
    if kind == "embedding":
        enc = tokenizer(["xin chào", "hello world, this is a longer sample sentence"],
                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    else:
        enc = tokenizer(["câu hỏi", "a question"], ["đoạn văn trả lời", "an answer passage that is longer"],
                        padding=True, truncation="longest_first", max_length=max_length, return_tensors="pt")

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *tensors):
            return self.inner(**dict(zip(names, tensors)), return_dict=False)[0]

    out_name = "last_hidden_state" if kind == "embedding" else "logits"
    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic[out_name] = {0: "batch", 1: "sequence"} if kind == "embedding" else {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            Wrapper(model), tuple(enc[n] for n in names), str(target),
            input_names=names, output_names=[out_name], dynamic_axes=dynamic,
            opset_version=opset, do_constant_folding=True,
        )


def quantize_int8(src: Path, dst: Path, reduce_range: bool, quantize_gather: bool) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    prepared = src
    try:  # tiền xử lý (shape inference) giúp lượng tử hóa chính xác hơn; không bắt buộc
        from onnxruntime.quantization.shape_inference import quant_pre_process

        prepared = src.with_suffix(".pre.onnx")
        quant_pre_process(str(src), str(prepared), skip_symbolic_shape=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (bỏ qua bước pre-process: {exc})")
        prepared = src

    # Mặc định chỉ lượng tử hóa MatMul/Gemm (các lớp attention + FFN) — phần quyết định tốc độ.
    # Bảng embedding từ (Gather) giữ fp32 để không làm lệch vector; bật --quantize-gather để giảm RAM.
    ops = ["MatMul", "Gemm"] + (["Gather"] if quantize_gather else [])
    quantize_dynamic(
        model_input=str(prepared), model_output=str(dst),
        weight_type=QuantType.QInt8, op_types_to_quantize=ops,
        per_channel=False, reduce_range=reduce_range,
    )
    if prepared != src and prepared.exists():
        prepared.unlink()


# --------------------------------------------------------------------------------------
# Tham chiếu PyTorch + đo
# --------------------------------------------------------------------------------------
class _TorchEmbedRef:
    """Tham chiếu: sentence-transformers nếu có (đúng như backend đang chạy), không thì mean pooling thủ công."""

    def __init__(self, name: str, model, tokenizer, max_length: int):
        self.max_length = max_length
        self.st = None
        try:
            from sentence_transformers import SentenceTransformer

            self.st = SentenceTransformer(name)
            self.max_length = int(getattr(self.st, "max_seq_length", None) or max_length)
            self.kind = "sentence-transformers"
        except Exception:  # noqa: BLE001
            self.model, self.tokenizer = model, tokenizer
            self.kind = "transformers + mean pooling"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.st is not None:
            return np.asarray(self.st.encode(list(texts), show_progress_bar=False), dtype=np.float32)
        import torch

        names = _input_names(self.tokenizer)
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        with torch.no_grad():
            hidden = self.model(**{n: enc[n] for n in names}, return_dict=False)[0]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).numpy().astype(np.float32)


class _TorchRerankRef:
    def __init__(self, name: str, model, tokenizer, max_length: int):
        self.st = None
        try:
            from sentence_transformers import CrossEncoder

            self.st = CrossEncoder(name, max_length=max_length)
            self.kind = "sentence-transformers"
        except Exception:  # noqa: BLE001
            self.model, self.tokenizer, self.max_length = model, tokenizer, max_length
            self.kind = "transformers"

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        if self.st is not None:
            return np.asarray(self.st.predict(list(pairs), show_progress_bar=False), dtype=np.float32)
        import torch

        names = _input_names(self.tokenizer)
        enc = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                             truncation="longest_first", max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**{n: enc[n] for n in names}, return_dict=False)[0].numpy()
        return (1 / (1 + np.exp(-logits[:, 0])) if logits.shape[1] == 1 else logits[:, -1]).astype(np.float32)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-x)).astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _median_ms(fn, repeat: int = 5) -> float:
    fn()  # khởi động
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def verify_embedding(ref: _TorchEmbedRef, runtime: onnx_models.OnnxEmbedder) -> dict:
    want = ref.encode(SAMPLE_TEXTS)
    got = runtime.encode(SAMPLE_TEXTS)
    cos = _cosine_rows(want, got)
    batch = SAMPLE_TEXTS[:8]
    return {
        "reference": ref.kind,
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "ms_reference": _median_ms(lambda: ref.encode(batch)),
        "ms_onnx": _median_ms(lambda: runtime.encode(batch)),
    }


def verify_reranker(ref: _TorchRerankRef, runtime: onnx_models.OnnxCrossEncoder) -> dict:
    rhos, top1, diffs = [], [], []
    for query, passages in RERANK_SETS:
        pairs = [(query, p) for p in passages]
        want, got = ref.predict(pairs), runtime.predict(pairs)
        rhos.append(_spearman(want, got))
        top1.append(int(np.argmax(want) == np.argmax(got)))
        diffs.append(float(np.abs(want - got).max()))
    pairs = [(RERANK_SETS[0][0], p) for p in RERANK_SETS[0][1]] * 3  # ~18 cặp, gần với rerank_top_n
    return {
        "reference": ref.kind,
        "spearman_mean": float(np.mean(rhos)),
        "top1_agreement": float(np.mean(top1)),
        "score_max_abs_diff": float(max(diffs)),
        "ms_reference": _median_ms(lambda: ref.predict(pairs)),
        "ms_onnx": _median_ms(lambda: runtime.predict(pairs)),
    }


# --------------------------------------------------------------------------------------
def process(kind: str, name: str, out_root: Path, args) -> bool:
    print(f"\n=== {kind}: {name} ===")
    max_length = args.embedding_max_length if kind == "embedding" else args.reranker_max_length
    model, tokenizer = _load_hf(name, kind)
    ref = (_TorchEmbedRef if kind == "embedding" else _TorchRerankRef)(name, model, tokenizer, max_length)
    if kind == "embedding":
        max_length = ref.max_length  # theo cấu hình của sentence-transformers (128 với mpnet-multilingual)

    directory = out_root / onnx_models.safe_model_dirname(name)
    directory.mkdir(parents=True, exist_ok=True)
    fp32 = onnx_models.model_file(directory, "fp32")
    int8 = onnx_models.model_file(directory, "int8")

    print("  Xuất ONNX fp32 …")
    export_fp32(model, tokenizer, kind, fp32, max_length, args.opset)
    print("  Lượng tử hóa INT8 …")
    quantize_int8(fp32, int8, reduce_range=not args.no_reduce_range, quantize_gather=args.quantize_gather)
    tokenizer.save_pretrained(str(directory))

    meta = {"kind": kind, "source_model": name, "max_length": max_length,
            "reduce_range": not args.no_reduce_range, "quantize_gather": args.quantize_gather}
    (directory / onnx_models.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cls = onnx_models.OnnxEmbedder if kind == "embedding" else onnx_models.OnnxCrossEncoder
    runtime = cls(directory, precision="int8", threads=args.threads)
    report = (verify_embedding if kind == "embedding" else verify_reranker)(ref, runtime)
    report["size_mb_int8"] = round(int8.stat().st_size / 1e6, 1)
    report["size_mb_fp32"] = round(fp32.stat().st_size / 1e6, 1)

    if kind == "embedding":
        ok = report["cosine_mean"] >= args.min_cosine and report["cosine_min"] >= args.min_cosine - 0.03
    else:
        ok = report["spearman_mean"] >= args.min_spearman and report["top1_agreement"] >= args.min_top1
    report["passed"] = bool(ok)
    meta["parity"] = report
    (directory / onnx_models.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for key, value in report.items():
        print(f"  {key:>16}: {round(value, 4) if isinstance(value, float) else value}")
    speedup = report["ms_reference"] / max(report["ms_onnx"], 1e-9)
    print(f"  → tốc độ ONNX INT8 so với PyTorch: x{speedup:.2f}")
    if not args.keep_fp32:
        fp32.unlink(missing_ok=True)
        data = fp32.with_name(fp32.name + ".data")
        data.unlink(missing_ok=True)
    print(f"  {'ĐẠT' if ok else 'KHÔNG ĐẠT'} ngưỡng chất lượng.  Thư mục: {directory}")
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", choices=["embedding", "reranker", "both"], default="both")
    parser.add_argument("--embedding-model", default=settings.rag_embedding_model)
    parser.add_argument("--reranker-model", default=settings.rag_reranker_model)
    parser.add_argument("--out", default=settings.rag_onnx_dir)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-reduce-range", action="store_true",
                        help="Tắt reduce_range (chỉ nên tắt trên CPU có AVX512-VNNI/AMX).")
    parser.add_argument("--quantize-gather", action="store_true",
                        help="Lượng tử hóa cả bảng embedding từ (giảm RAM, nhưng vector lệch nhiều hơn).")
    parser.add_argument("--keep-fp32", action="store_true", help="Giữ file ONNX fp32 (nặng, ~1 GB cho mpnet).")
    parser.add_argument("--min-cosine", type=float, default=0.97)
    parser.add_argument("--min-spearman", type=float, default=0.85)
    parser.add_argument("--min-top1", type=float, default=0.66)
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    results = []
    if args.which in ("embedding", "both"):
        results.append(process("embedding", args.embedding_model, out_root, args))
    if args.which in ("reranker", "both"):
        results.append(process("reranker", args.reranker_model, out_root, args))

    if all(results):
        print("\nXong. Khởi động lại backend để dùng bản INT8 (docker compose restart backend-ai celery-worker).")
        print("Nên upload lại tài liệu để vector trong Qdrant cùng bản embedding với truy vấn.")
        return 0
    print("\nCẢNH BÁO: có model không đạt ngưỡng chất lượng. KHÔNG nên dùng bản INT8 đó "
          "(xóa thư mục ở trên hoặc đặt RAG_ONNX_ENABLED=false).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
