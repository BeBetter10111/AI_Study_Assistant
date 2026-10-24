"""Automatic RAG evaluation with RAGAS.

Usage (from backend-ai/, backend + Qdrant + Ollama running, documents uploaded):

    python -m app.rag.evaluation eval/eval_set.jsonl --tag hybrid-a0.7
    python -m app.rag.evaluation eval/eval_set.jsonl --alpha 1.0 --tag dense-only
    python -m app.rag.evaluation eval/eval_set.jsonl --no-rerank --tag no-rerank

Dataset: JSON-lines (or a JSON list). One object per question:
    question      (required)  alias: user_input
    ground_truth  (optional)  alias: reference — enables context_recall / factual_correctness
    document_id   (optional)  restrict retrieval to one uploaded document
    response, retrieved_contexts (optional) — supply both to judge answers produced
                  elsewhere; the RAG pipeline is then skipped for that row.

For every row the *real* pipeline runs (hybrid search -> prompt -> LLM), then RAGAS
scores it.  A JSON report (summary + per-question rows + retrieval config) and a CSV
are written to RAGAS_OUTPUT_DIR so different retrieval settings can be compared.

Heavy imports (ragas, langchain, the app pipeline) are lazy: the pure helpers below
are importable and unit-testable without them.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings

logger = logging.getLogger(__name__)

# Friendly name -> what it measures
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "factual_correctness")
REFERENCE_REQUIRED = frozenset({"context_recall", "factual_correctness"})
DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_NO_ANSWER = "No relevant content found in the uploaded document."


# ----------------------------------------------------------------- pure helpers
def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Read and normalise an evaluation set (.jsonl or .json list)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"{path} is empty.")
    if path.suffix.lower() == ".json":
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list of objects.")
    else:
        records = []
        for n, line in enumerate(raw.splitlines(), start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg}).") from exc

    items: List[Dict[str, Any]] = []
    for n, rec in enumerate(records, start=1):
        question = str(rec.get("question") or rec.get("user_input") or "").strip()
        if not question:
            raise ValueError(f"{path}: row {n} has no 'question'.")
        reference = str(rec.get("ground_truth") or rec.get("reference") or "").strip()
        contexts = rec.get("retrieved_contexts")
        if contexts is not None and not isinstance(contexts, list):
            raise ValueError(f"{path}: row {n}: 'retrieved_contexts' must be a list of strings.")
        items.append({
            "question": question,
            "reference": reference or None,
            "document_id": rec.get("document_id") or None,
            "response": rec.get("response"),
            "retrieved_contexts": [str(c) for c in contexts] if contexts is not None else None,
        })
    return items


def select_metric_names(requested: Optional[Sequence[str]], has_reference: bool) -> List[str]:
    """Validate requested metrics; default set depends on whether references exist."""
    if not requested:
        return [m for m in DEFAULT_METRICS if has_reference or m not in REFERENCE_REQUIRED]
    unknown = [m for m in requested if m not in METRIC_NAMES]
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown)}. Choices: {', '.join(METRIC_NAMES)}.")
    missing = [m for m in requested if m in REFERENCE_REQUIRED and not has_reference]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} need 'ground_truth' for every row of the evaluation set."
        )
    return list(dict.fromkeys(requested))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def summarize(rows: Sequence[Dict[str, Any]], metric_cols: Sequence[str]) -> Dict[str, Any]:
    """Mean per metric over rows with a valid score; also count failed judge calls."""
    out: Dict[str, Any] = {}
    for col in metric_cols:
        vals = [r[col] for r in rows if _is_number(r.get(col))]
        out[col] = {
            "mean": round(statistics.fmean(vals), 4) if vals else None,
            "scored": len(vals),
            "failed": len(rows) - len(vals),
        }
    return out


def latency_stats(values_ms: Sequence[float]) -> Optional[Dict[str, float]]:
    vals = sorted(v for v in values_ms if _is_number(v))
    if not vals:
        return None
    p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)]
    return {"mean_ms": round(statistics.fmean(vals), 1), "p95_ms": round(p95, 1)}


def write_report(report: Dict[str, Any], out_dir: str | Path, tag: str = "") -> Path:
    """Write <timestamp>[_tag].json and .csv; return the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(c if c.isalnum() or c in "-_." else "-" for c in tag).strip("-")
    stem = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{safe_tag}" if safe_tag else "")
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = report["rows"]
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with (out_dir / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                                 for k, v in r.items()})
    return json_path


# --------------------------------------------------------------- RAGAS plumbing
def build_judge_llm():
    """RAGAS-wrapped judge LLM (Ollama by default, or any OpenAI-compatible server)."""
    from ragas.llms import LangchainLLMWrapper

    provider = settings.ragas_judge_provider.strip().lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.ragas_judge_model or settings.ollama_model_name,
            base_url=settings.ragas_judge_base_url or None,
            api_key=settings.ragas_judge_api_key,
            temperature=0,
            timeout=settings.ragas_timeout_s,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            model=settings.ragas_judge_model or settings.ollama_model_name,
            base_url=(settings.ragas_judge_base_url or settings.ollama_base_url).rstrip("/"),
            temperature=0,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.ragas_timeout_s,
        )
    else:
        raise ValueError("RAGAS_JUDGE_PROVIDER must be 'ollama' or 'openai'.")
    return LangchainLLMWrapper(llm)


def build_embeddings():
    """RAGAS-wrapped embeddings: reuse the app's multilingual model (no extra download)."""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from app.rag.vector_store import _get_embedding_model

    class _AppEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return _get_embedding_model().encode(texts, show_progress_bar=False).tolist()

        def embed_query(self, text: str) -> List[float]:
            return _get_embedding_model().encode(text).tolist()

    return LangchainEmbeddingsWrapper(_AppEmbeddings())


def build_metrics(names: Sequence[str], has_reference: bool) -> List[Any]:
    from ragas import metrics as m

    factories = {
        "faithfulness": m.Faithfulness,
        "answer_relevancy": m.ResponseRelevancy,
        "context_precision": (m.LLMContextPrecisionWithReference if has_reference
                              else m.LLMContextPrecisionWithoutReference),
        "context_recall": m.LLMContextRecall,
        "factual_correctness": m.FactualCorrectness,
    }
    return [factories[n]() for n in names]


# -------------------------------------------------------------- pipeline runner
def run_pipeline(item: Dict[str, Any], top_k: int, alpha: Optional[float]) -> Dict[str, Any]:
    """Run the production RAG path (hybrid search -> prompt -> LLM) for one question."""
    from app.api.endpoints.chat import _build_prompts, _classify_intent
    from app.core.llm_engine import llm_engine
    from app.rag.vector_store import vector_store

    question = item["question"]
    intent = _classify_intent(question)
    if intent in ("identity", "quiz_generation"):  # these skip RAG in production
        intent = "qa"

    t0 = time.perf_counter()
    hits = vector_store.search_similar_chunks(
        query=question, limit=top_k, document_id=item.get("document_id"), alpha=alpha
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000
    if not hits:
        return {"response": _NO_ANSWER, "retrieved_contexts": [],
                "retrieval_ms": retrieval_ms, "generation_ms": 0.0}

    system_prompt, user_prompt = _build_prompts(intent, hits, question)
    t1 = time.perf_counter()
    answer = llm_engine.generate_rag_response(prompt=user_prompt, system_prompt=system_prompt)
    generation_ms = (time.perf_counter() - t1) * 1000
    return {"response": answer, "retrieved_contexts": [h["content"] for h in hits],
            "retrieval_ms": retrieval_ms, "generation_ms": generation_ms}


def evaluate_samples(
    samples: List[Dict[str, Any]], metric_names: Sequence[str], has_reference: bool,
    llm: Any = None, embeddings: Any = None,
) -> List[Dict[str, Any]]:
    """Score samples with RAGAS; returns one dict of metric scores per sample."""
    from ragas import EvaluationDataset, RunConfig, evaluate

    metrics = build_metrics(metric_names, has_reference)
    data = []
    for s in samples:
        row = {"user_input": s["question"], "response": s["response"],
               "retrieved_contexts": s["retrieved_contexts"]}
        if has_reference:
            row["reference"] = s["reference"]
        data.append(row)

    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=metrics,
        llm=llm or build_judge_llm(),
        embeddings=embeddings or build_embeddings(),
        run_config=RunConfig(timeout=int(settings.ragas_timeout_s), max_workers=settings.ragas_max_workers),
        raise_exceptions=False,
        show_progress=True,
    )
    df = result.to_pandas()
    cols = [m.name for m in metrics]
    return [{c: (None if c not in df.columns or df[c].iloc[i] != df[c].iloc[i] else float(df[c].iloc[i]))
             for c in cols} for i in range(len(samples))]


def run_evaluation(
    items: List[Dict[str, Any]], metric_names: Optional[Sequence[str]] = None,
    top_k: int = 4, alpha: Optional[float] = None, tag: str = "",
) -> Dict[str, Any]:
    has_reference = all(it["reference"] for it in items)
    names = select_metric_names(metric_names, has_reference)

    samples: List[Dict[str, Any]] = []
    for n, item in enumerate(items, start=1):
        if item["response"] is not None and item["retrieved_contexts"] is not None:
            out = {"response": item["response"], "retrieved_contexts": item["retrieved_contexts"],
                   "retrieval_ms": None, "generation_ms": None}
        else:
            logger.info(f"[{n}/{len(items)}] running pipeline: {item['question'][:70]}")
            out = run_pipeline(item, top_k, alpha)
        samples.append({**item, **out})

    # Rows with nothing retrieved cannot be judged — report them as retrieval failures.
    judged_idx = [i for i, s in enumerate(samples) if s["retrieved_contexts"]]
    scores: Dict[int, Dict[str, Optional[float]]] = {}
    if judged_idx:
        for i, sc in zip(judged_idx, evaluate_samples([samples[i] for i in judged_idx], names, has_reference)):
            scores[i] = sc

    metric_cols = list(next(iter(scores.values())).keys()) if scores else []
    rows = []
    for i, s in enumerate(samples):
        rows.append({
            "question": s["question"], "document_id": s["document_id"], "reference": s["reference"],
            "response": s["response"], "retrieved_contexts": s["retrieved_contexts"],
            "retrieval_ms": s["retrieval_ms"], "generation_ms": s["generation_ms"],
            **{c: scores.get(i, {}).get(c) for c in metric_cols},
        })

    return {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"), "tag": tag, "n_questions": len(items),
            "n_without_retrieval": len(items) - len(judged_idx),
            "llm_backend": settings.llm_backend, "answer_model": settings.ollama_model_name,
            "judge": {"provider": settings.ragas_judge_provider,
                      "model": settings.ragas_judge_model or settings.ollama_model_name},
            "retrieval": {"top_k": top_k, "fetch_k": settings.rag_fetch_k,
                          "alpha": settings.rag_hybrid_alpha if alpha is None else alpha,
                          "rrf_k": settings.rag_rrf_k, "rerank_enabled": settings.rag_rerank_enabled,
                          "reranker": settings.rag_reranker_model, "rerank_weight": settings.rag_rerank_weight},
        },
        "summary": summarize(rows, metric_cols),
        "latency": {"retrieval": latency_stats([r["retrieval_ms"] for r in rows if r["retrieval_ms"] is not None]),
                    "generation": latency_stats([r["generation_ms"] for r in rows if r["generation_ms"]])},
        "rows": rows,
    }


# ------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    ap.add_argument("dataset", help="Path to .jsonl / .json evaluation set")
    ap.add_argument("--document-id", help="Default document_id for rows that have none")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--alpha", type=float, help="Dense weight override (1.0 dense only, 0.0 BM25 only)")
    ap.add_argument("--no-rerank", action="store_true", help="Disable the cross-encoder for this run")
    ap.add_argument("--metrics", nargs="+", choices=METRIC_NAMES, help="Default: all that apply")
    ap.add_argument("--limit", type=int, help="Only evaluate the first N questions")
    ap.add_argument("--tag", default="", help="Label for the report file name")
    ap.add_argument("--out-dir", default=settings.ragas_output_dir)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.no_rerank:
        settings.rag_rerank_enabled = False

    items = load_dataset(args.dataset)[: args.limit]
    for it in items:
        it["document_id"] = it["document_id"] or args.document_id
    report = run_evaluation(items, args.metrics, args.top_k, args.alpha, args.tag)
    path = write_report(report, args.out_dir, args.tag)

    print(f"\nRAGAS report ({report['meta']['n_questions']} questions) -> {path}")
    for name, s in report["summary"].items():
        mean = "n/a" if s["mean"] is None else f"{s['mean']:.3f}"
        print(f"  {name:<40} {mean}   (scored {s['scored']}, failed {s['failed']})")
    for part, st in report["latency"].items():
        if st:
            print(f"  {part} latency: mean {st['mean_ms']:.0f} ms, p95 {st['p95_ms']:.0f} ms")
    if report["meta"]["n_without_retrieval"]:
        print(f"  WARNING: {report['meta']['n_without_retrieval']} question(s) retrieved nothing "
              "(check document_id / that documents are embedded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
