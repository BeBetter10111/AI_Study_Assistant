"""Kiểm tra end-to-end ONNX INT8 trên model Transformer NHỎ ngẫu nhiên (không cần tải gì từ mạng).

Chạy ĐỘC LẬP (không qua conftest vì conftest giả lập torch):  python tests/onnx_e2e_check.py <arch>
với <arch> là "bert" (3 input, có token_type_ids) hoặc "distilbert" (2 input, như XLM-R/mpnet).
In ra một dòng JSON kết quả; tests/test_onnx_quantization.py chạy file này trong subprocess.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types  # noqa: E402

try:  # chỉ để import được app.rag.vector_store trong môi trường tối giản (không dùng đến chunking ở đây)
    import langchain_text_splitters  # noqa: F401
except ImportError:
    sys.modules["langchain_text_splitters"] = types.SimpleNamespace(RecursiveCharacterTextSplitter=object)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertModel,
    DistilBertTokenizerFast,
)

from app.config import settings  # noqa: E402
from app.rag import onnx_models  # noqa: E402
from app.rag import vector_store as vs
from scripts import quantize_models as qm  # noqa: E402

torch.manual_seed(0)
arch = sys.argv[1] if len(sys.argv) > 1 else "bert"
work = Path(tempfile.mkdtemp(prefix="onnx_e2e_"))
result = {"arch": arch}


def build_tokenizer(cls):
    chars = sorted({c for t in qm.SAMPLE_TEXTS + [p for _, ps in qm.RERANK_SETS for p in ps]
                    + [q for q, _ in qm.RERANK_SETS] for c in t.lower() if not c.isspace()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    path = work / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    return cls(vocab_file=str(path), do_lower_case=True, strip_accents=False), len(vocab)


def make(kind):
    if arch == "bert":
        tok, n = build_tokenizer(BertTokenizerFast)
        cfg = BertConfig(vocab_size=n, hidden_size=48, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=96, max_position_embeddings=128, num_labels=1)
        model = BertModel(cfg) if kind == "embedding" else BertForSequenceClassification(cfg)
    else:
        tok, n = build_tokenizer(DistilBertTokenizerFast)
        cfg = DistilBertConfig(vocab_size=n, dim=48, n_layers=2, n_heads=2, hidden_dim=96,
                               max_position_embeddings=128, num_labels=1)
        model = DistilBertModel(cfg) if kind == "embedding" else DistilBertForSequenceClassification(cfg)
    path = work / f"tiny-{kind}"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


emb_name, rr_name = make("embedding"), make("reranker")

out = work / "onnx"
code = qm.main([
    "--embedding-model", emb_name, "--reranker-model", rr_name, "--out", str(out),
    "--embedding-max-length", "64", "--reranker-max-length", "96",
    "--min-cosine", "0.90", "--min-spearman", "-1", "--min-top1", "0",
])
result["main_exit_code"] = code

# --- file layout + proof the int8 graph is really quantized ---------------------------------
import onnx  # noqa: E402

edir = out / onnx_models.safe_model_dirname(emb_name)
rdir = out / onnx_models.safe_model_dirname(rr_name)
result["files"] = sorted(p.name for p in edir.iterdir())
ops = {n.op_type for n in onnx.load(str(edir / "model.int8.onnx")).graph.node}
result["int8_ops"] = sorted(o for o in ops if "Integer" in o or "Quantize" in o)
result["fp32_removed"] = not (edir / "model.fp32.onnx").exists()
result["onnx_inputs"] = [i.name for i in onnx.load(str(edir / "model.int8.onnx")).graph.input]
meta = onnx_models.read_meta(edir)
result["embed_parity"] = {k: meta["parity"][k] for k in ("cosine_mean", "cosine_min", "passed")}
result["rerank_parity"] = {k: onnx_models.read_meta(rdir)["parity"][k]
                           for k in ("spearman_mean", "top1_agreement", "score_max_abs_diff")}

# --- runtime API: shapes, batching invariance, single vs list --------------------------------
emb = onnx_models.OnnxEmbedder(edir, precision="int8")
texts = qm.SAMPLE_TEXTS
one = emb.encode(texts[0])
many = emb.encode(texts)
bs1 = emb.encode(texts, batch_size=1)
bs32 = emb.encode(texts, batch_size=32)
result["embed_shapes"] = [list(one.shape), list(many.shape)]
result["embed_single_vs_batch_maxdiff"] = float(np.abs(one - many[0]).max())
result["embed_batchsize_maxdiff"] = float(np.abs(bs1 - bs32).max())
result["embed_order_ok"] = bool(np.allclose(emb.encode(list(reversed(texts))), many[::-1], atol=1e-4))

ce = onnx_models.OnnxCrossEncoder(rdir, precision="int8")
pairs = [(qm.RERANK_SETS[0][0], p) for p in qm.RERANK_SETS[0][1]]
scores = ce.predict(pairs)
result["rerank_range_ok"] = bool(((scores >= 0) & (scores <= 1)).all() and len(scores) == len(pairs))
result["rerank_order_ok"] = bool(np.allclose(ce.predict(pairs[::-1]), scores[::-1], atol=1e-4))
result["rerank_empty"] = list(ce.predict([]).shape)

# --- nhiều thread dùng chung một model (như threadpool của FastAPI): kết quả phải giống tuần tự ----
import threading  # noqa: E402

errors, outs = [], {}
# Lượng tử hóa động tính hệ số co giãn theo cả batch nên kết quả phụ thuộc nhẹ vào các câu đi cùng;
# vì vậy so với kết quả tuần tự CÙNG thành phần batch (một câu / cùng danh sách cặp).
single_ref = {i: emb.encode(texts[i % len(texts)]) for i in range(8)}


def worker(i):
    try:
        for _ in range(15):
            outs[i] = (emb.encode(texts[i % len(texts)]), ce.predict(pairs))
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
result["concurrent_errors"] = errors
result["concurrent_consistent"] = bool(
    all(np.allclose(outs[i][0], single_ref[i], atol=1e-6) and np.allclose(outs[i][1], scores, atol=1e-6)
        for i in outs) and len(outs) == 8)

# --- vector_store: dùng ONNX khi có file, tự quay về PyTorch khi thiếu -----------------------
settings.rag_onnx_enabled, settings.rag_onnx_dir, settings.rag_onnx_precision = True, str(out), "int8"
settings.rag_embedding_model, settings.rag_reranker_model, settings.rag_rerank_enabled = emb_name, rr_name, True
vs._embedding_model = vs._reranker = None
model = vs._get_embedding_model()
result["vs_embedding_backend"] = vs._embedding_backend
result["vs_is_onnx"] = isinstance(model, onnx_models.OnnxEmbedder)
vs._get_reranker()
result["vs_reranker_backend"] = vs._reranker_backend
result["vs_fingerprint_onnx"] = vs.model_fingerprint()

calls = {"n": 0}
orig = model.encode
model.encode = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
v1, v2 = vs.embed_query("stack là gì"), vs.embed_query("stack là gì")
result["embed_query_cached_calls"] = calls["n"]
v1[0] += 100  # sửa bản sao không được làm hỏng cache
result["embed_query_copy_safe"] = bool(np.allclose(vs.embed_query("stack là gì"), v2))

settings.rag_onnx_dir = str(work / "does-not-exist")  # thiếu file => quay về PyTorch (hành vi cũ)
vs._embedding_model = None
try:
    class _ST:  # SentenceTransformer giả, để không phải tải model thật
        def __init__(self, *a, **k):
            pass

        def encode(self, *a, **k):
            return np.zeros(4, dtype=np.float32)

    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_ST)
    fallback = vs._get_embedding_model()
    result["vs_fallback_backend"] = vs._embedding_backend
    result["vs_fallback_is_torch_class"] = isinstance(fallback, _ST)
finally:
    pass

print("RESULT_JSON=" + json.dumps(result))
