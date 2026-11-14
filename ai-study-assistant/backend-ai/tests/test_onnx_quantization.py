"""ONNX INT8 end-to-end: xuất + lượng tử hóa + chạy thử trên model Transformer nhỏ ngẫu nhiên.

Cần torch/onnx/onnxruntime/transformers THẬT, trong khi conftest giả lập torch trong tiến trình pytest,
nên kiểm tra thật chạy ở subprocess (tests/onnx_e2e_check.py). Tự bỏ qua nếu thiếu thư viện (CI nhẹ).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _have_ml_stack() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch, onnx, onnxruntime, transformers, tokenizers"],
        capture_output=True, cwd=ROOT,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _have_ml_stack(), reason="needs real torch + onnx + onnxruntime + transformers")


def _run(arch: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "tests/onnx_e2e_check.py", arch],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:] + proc.stdout[-2000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON="))
    return json.loads(line[len("RESULT_JSON="):])


@pytest.mark.parametrize("arch", ["bert", "distilbert"])  # có / không có input token_type_ids
def test_export_quantize_and_run(arch):
    r = _run(arch)
    assert r["main_exit_code"] == 0
    assert "model.int8.onnx" in r["files"] and "meta.json" in r["files"] and "tokenizer.json" in r["files"]
    assert r["fp32_removed"] is True
    assert "MatMulInteger" in r["int8_ops"]                       # đồ thị thật sự đã được lượng tử hóa
    assert ("token_type_ids" in r["onnx_inputs"]) == (arch == "bert")

    assert r["embed_parity"]["passed"] and r["embed_parity"]["cosine_mean"] > 0.95
    assert r["rerank_parity"]["score_max_abs_diff"] < 0.05

    assert r["embed_shapes"][0] == [48] and r["embed_shapes"][1] == [10, 48]   # str -> 1-D, list -> 2-D
    assert r["embed_single_vs_batch_maxdiff"] < 1e-3                          # không phụ thuộc padding/batch
    assert r["embed_batchsize_maxdiff"] < 1e-3
    assert r["embed_order_ok"] and r["rerank_order_ok"]                        # kết quả đúng thứ tự đầu vào
    assert r["rerank_range_ok"] and r["rerank_empty"] == [0]

    assert r["concurrent_errors"] == [] and r["concurrent_consistent"]        # an toàn khi nhiều thread dùng chung

    assert (r["vs_embedding_backend"], r["vs_reranker_backend"]) == ("onnx-int8", "onnx-int8")
    assert r["embed_query_cached_calls"] == 1 and r["embed_query_copy_safe"]
    assert r["vs_fallback_backend"] == "torch" and r["vs_fallback_is_torch_class"]   # thiếu file => PyTorch
