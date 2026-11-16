"""Vector store — multilingual embeddings, true hybrid (dense + BM25) search, re-ranking.

Pipeline of ``search_similar_chunks``
-------------------------------------
1. **Dense leg**  — multilingual mpnet embedding (ONNX INT8 nếu đã lượng tử hóa, ngược lại
   PyTorch), cosine search in Qdrant.
2. **Sparse leg** — BM25 term weights stored as a Qdrant sparse vector (IDF is
   computed by Qdrant over the whole collection), searched *independently*
   so exact keywords / course codes / function names are retrieved even when
   the dense model misses them.  Both legs run in one ``query_batch_points``
   round-trip and share the same ``document_id`` filter.
3. **Fusion**     — weighted Reciprocal Rank Fusion (``alpha`` = dense weight).
4. **Re-ranking** — multilingual cross-encoder (ONNX INT8 hoặc PyTorch) on the top candidates;
   its score is *blended* with the fusion score (not a hard override).
5. **Clean-up**   — drop summary chunks that duplicate a returned detail chunk.

Schema (breaking change)
------------------------
The collection now uses named vectors ``dense`` + sparse ``bm25``.  A
collection created by the previous version must be deleted and its documents
re-uploaded (``scripts/bulk_upload.py``).
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings
from app.rag import hybrid
from app.rag.chunking import ChunkData

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "bm25"

# Lazy singletons for heavy models (loaded only on first use)
_embedding_model = None
_reranker = None
_reranker_failed = False
# "onnx-int8" | "onnx-fp32" | "torch" — used for logging and the semantic-cache key.
_embedding_backend = "unloaded"
_reranker_backend = "unloaded"

_load_lock = threading.Lock()
_query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_query_cache_lock = threading.Lock()


class CollectionSchemaError(RuntimeError):
    """The Qdrant collection does not match what this code version expects."""


def _try_onnx(model_name: str, kind: str):
    """Tải bản ONNX đã lượng tử hóa nếu có; trả None để caller dùng PyTorch."""
    if not settings.rag_onnx_enabled:
        return None
    from app.rag import onnx_models

    precision = settings.rag_onnx_precision.strip().lower()
    if precision not in ("int8", "fp32"):
        logger.warning(f"RAG_ONNX_PRECISION='{precision}' is invalid (use int8 or fp32) — using PyTorch.")
        return None
    directory = onnx_models.find_model_dir(settings.rag_onnx_dir, model_name, precision)
    if directory is None:
        logger.info(
            f"No ONNX {precision} {kind} for {model_name} under '{settings.rag_onnx_dir}' — using PyTorch. "
            f"Create it with: python -m scripts.quantize_models"
        )
        return None
    try:
        cls = onnx_models.OnnxEmbedder if kind == "embedding" else onnx_models.OnnxCrossEncoder
        return cls(directory, precision=precision, threads=settings.rag_onnx_threads)
    except Exception as exc:  # Corrupt file / missing onnxruntime / missing tokenizer, etc.
        logger.warning(f"Could not load ONNX {kind} from {directory} — using PyTorch: {exc}")
        return None


def _get_embedding_model():
    global _embedding_model, _embedding_backend
    if _embedding_model is None:
        with _load_lock:
            if _embedding_model is None:
                name = settings.rag_embedding_model
                model = _try_onnx(name, "embedding")
                if model is not None:
                    _embedding_backend = f"onnx-{model.precision}"
                else:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading multilingual embedding model (PyTorch) …")
                    model = SentenceTransformer(name)
                    _embedding_backend = "torch"
                _embedding_model = model
                logger.info(f"Embedding model ready — backend: {_embedding_backend}.")
    return _embedding_model


def _get_reranker():
    """Return the cross-encoder, or None if disabled / unavailable."""
    global _reranker, _reranker_failed, _reranker_backend
    if not settings.rag_rerank_enabled or _reranker_failed:
        return None
    if _reranker is None:
        with _load_lock:
            if _reranker is None and not _reranker_failed:
                name = settings.rag_reranker_model
                model = _try_onnx(name, "reranker")
                if model is not None:
                    _reranker_backend = f"onnx-{model.precision}"
                else:
                    try:
                        from sentence_transformers import CrossEncoder
                        logger.info(f"Loading cross-encoder re-ranker {name} (PyTorch) …")
                        model = CrossEncoder(name, max_length=512)
                        _reranker_backend = "torch"
                    except Exception as exc:
                        logger.warning(
                            f"Could not load re-ranker — falling back to fusion-only ranking: {exc}"
                        )
                        _reranker_failed = True
                        return None
                _reranker = model
                logger.info(f"Re-ranker ready — backend: {_reranker_backend}.")
    return _reranker


def model_fingerprint() -> str:
    """Định danh cấu hình model đang chạy — đổi model/backend thì cache ngữ nghĩa tự vô hiệu."""
    _get_embedding_model()
    rerank = _reranker_backend if _get_reranker() is not None else "off"
    return f"{settings.rag_embedding_model}@{_embedding_backend}|{settings.rag_reranker_model}@{rerank}"


def embed_query(text: str) -> np.ndarray:
    """Embedding của MỘT câu truy vấn, có LRU cache.

    Dùng chung cho tìm kiếm dense và semantic cache nên mỗi câu hỏi chỉ phải embed một lần.
    Trả về bản sao (caller được phép sửa).
    """
    size = max(0, settings.rag_query_embed_cache_size)
    if size:
        with _query_cache_lock:
            hit = _query_cache.get(text)
            if hit is not None:
                _query_cache.move_to_end(text)
                return hit.copy()
    vector = np.asarray(_get_embedding_model().encode(text), dtype=np.float32)
    if size:
        with _query_cache_lock:
            _query_cache[text] = vector
            _query_cache.move_to_end(text)
            while len(_query_cache) > size:
                _query_cache.popitem(last=False)
    return vector.copy()


def _sparse_text(content: str, title: Optional[str]) -> str:
    """Titles carry strong keyword signal, so they are indexed with the body."""
    return f"{title}\n{content}" if title else content


def _title_of(payload: Dict[str, Any]) -> Optional[str]:
    return payload.get("slide_title") or payload.get("section_title")


# Domain metadata from ChunkData.extra persisted in the Qdrant payload (whitelist).
EXTRA_PAYLOAD_KEYS = ("content_format", "domain", "hierarchy_path", "article_key", "article", "clause")


def hit_from_payload(payload: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
    """Search-result dict (same shape for search hits and legal sibling expansion)."""
    hit = {
        "content": payload.get("content", ""),
        "document_id": payload.get("document_id", ""),
        "chunk_index": payload.get("chunk_index", 0),
        "page_number": payload.get("page_number"),
        "slide_number": payload.get("slide_number"),
        "slide_title": payload.get("slide_title"),
        "section_title": payload.get("section_title"),
        "is_summary": payload.get("is_summary", False),
        "score": score,
    }
    for key in EXTRA_PAYLOAD_KEYS:
        hit[key] = payload.get(key)
    return hit


def expanded_query_text(query: str, expansion_terms: Optional[List[str]]) -> str:
    """Query text used for retrieval when entity linking added synonyms (brand -> ingredient ...)."""
    terms = [t for t in (expansion_terms or []) if t and t.lower() not in query.lower()]
    return f"{query} ({'; '.join(terms)})" if terms else query


class VectorStoreManager:
    """Manages Qdrant collection, multilingual embeddings, hybrid search, re-ranking."""

    # Multilingual mpnet-base-v2 produces 768-dim vectors.
    VECTOR_SIZE = 768

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.qdrant_collection
        self._qdrant_url = settings.qdrant_url
        self._client = None
        self._collection_ready = False

    # Qdrant client (lazy)
    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self._qdrant_url)
        return self._client

    # Collection management
    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        self.init_collection()
        self._collection_ready = True

    def init_collection(self) -> None:
        try:
            from qdrant_client.http import models

            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_NAME: models.VectorParams(
                            size=self.VECTOR_SIZE, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        # IDF is computed by Qdrant over the whole collection.
                        SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
                try:  # speeds up the per-document filter used by every query
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:
                    logger.warning(f"Could not create payload index on document_id: {exc}")
                try:  # speeds up per-owner filter — critical for user data isolation
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="owner_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:
                    logger.warning(f"Could not create payload index on owner_id: {exc}")
                self._ensure_article_index()
                logger.info(f"Created Qdrant collection '{self.collection_name}' "
                            f"(dense={self.VECTOR_SIZE}, sparse={SPARSE_NAME}).")
                return

            params = self.client.get_collection(self.collection_name).config.params
            dense_cfg = params.vectors.get(DENSE_NAME) if isinstance(params.vectors, dict) else None
            has_sparse = bool(params.sparse_vectors) and SPARSE_NAME in params.sparse_vectors
            if dense_cfg is None or not has_sparse:
                raise CollectionSchemaError(
                    f"Qdrant collection '{self.collection_name}' uses the old dense-only schema. "
                    f"Delete the collection and re-upload your documents to enable hybrid search."
                )
            if dense_cfg.size != self.VECTOR_SIZE:
                raise CollectionSchemaError(
                    f"Qdrant collection '{self.collection_name}' has vector size {dense_cfg.size}, "
                    f"but the embedding model produces {self.VECTOR_SIZE}. "
                    f"Delete the collection and re-upload your documents."
                )
            logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
            self._ensure_article_index()
        except Exception as e:
            logger.error(f"Error initialising Qdrant collection: {e}")
            raise

    def _ensure_article_index(self) -> None:
        """Keyword index for legal sibling-clause lookups (idempotent, best effort)."""
        try:
            from qdrant_client.http import models

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="article_key",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Payload index on article_key not created: {exc}")

    # Upsert — accepts ChunkData objects with rich metadata
    def upsert_chunks(
        self,
        chunks: List[ChunkData],
        document_id: str,
        start_chunk_idx: int = 0,
        owner_id: Optional[str] = None,
    ) -> bool:
        """Embed chunks (dense + BM25 sparse) and upsert into Qdrant.

        :param chunks: List of ``ChunkData`` from ingestion.py
        :param document_id: Parent document UUID
        :param start_chunk_idx: Offset for chunk_index (for incremental ingestion)
        :param owner_id: UUID of the user who owns this document.
                         Stored in every point payload so that searches can be
                         scoped to the calling user (prevents cross-user data leakage).
        :return: True on success
        """
        if not chunks:
            logger.warning("Empty chunks list — skipping upsert.")
            return False

        self._ensure_collection()

        try:
            from qdrant_client.http import models as qdrant_models

            texts = [c.content for c in chunks]
            embeddings = _get_embedding_model().encode(texts, show_progress_bar=False)

            points = []
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                chunk_index = start_chunk_idx + idx
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}")
                )

                payload: Dict[str, Any] = {
                    "document_id": str(document_id),
                    "chunk_index": chunk_index,
                    "content": chunk.content,
                    "is_summary": chunk.extra.get("is_summary", False),
                }
                # Security: store owner_id so search can filter by user.
                # This prevents any user from reading another user's document
                # chunks by guessing/iterating document UUIDs.
                if owner_id:
                    payload["owner_id"] = str(owner_id)
                # Persist structural metadata — only include keys that have values
                if chunk.page_number is not None:
                    payload["page_number"] = chunk.page_number
                if chunk.slide_number is not None:
                    payload["slide_number"] = chunk.slide_number
                if chunk.slide_title:
                    payload["slide_title"] = chunk.slide_title
                if chunk.section_title:
                    payload["section_title"] = chunk.section_title
                for key in EXTRA_PAYLOAD_KEYS:
                    value = chunk.extra.get(key)
                    if value is not None and value != "":
                        payload[key] = value

                vectors: Dict[str, Any] = {DENSE_NAME: vector.tolist()}
                s_idx, s_val = hybrid.sparse_document_vector(
                    _sparse_text(chunk.content, _title_of(payload)),
                    bigrams=settings.rag_sparse_bigrams,
                )
                if s_idx:
                    vectors[SPARSE_NAME] = qdrant_models.SparseVector(indices=s_idx, values=s_val)

                points.append(
                    qdrant_models.PointStruct(id=point_id, vector=vectors, payload=payload)
                )

            batch_size = 20
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[i : i + batch_size],
                )
            logger.info(f"Upserted {len(points)} vectors for document {document_id}.")
            return True

        except Exception as e:
            logger.error(f"Error upserting vectors into Qdrant: {e}")
            raise

    # Hybrid search (dense + BM25 sparse) -> RRF -> re-rank
    def delete_document(self, document_id: str, owner_id: Optional[str] = None) -> None:
        """Xóa MỌI điểm (chunk dense + BM25 + chunk tóm tắt) của một tài liệu khỏi Qdrant.

        Lọc theo cả ``owner_id`` (nếu có) để một id tài liệu của người khác không bao giờ bị xóa nhầm.
        Ném lỗi nếu Qdrant lỗi — caller phải dừng việc xóa tài liệu để không bỏ lại vector mồ côi.
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_collection()
        must = [qdrant_models.FieldCondition(
            key="document_id", match=qdrant_models.MatchValue(value=str(document_id)))]
        if owner_id:
            must.append(qdrant_models.FieldCondition(
                key="owner_id", match=qdrant_models.MatchValue(value=str(owner_id))))
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(filter=qdrant_models.Filter(must=must)),
        )

    def search_similar_chunks(
        self,
        query: str,
        limit: int = 4,
        document_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        fetch_k: Optional[int] = None,
        alpha: Optional[float] = None,
        expansion_terms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid dense + BM25 retrieval, RRF fusion, cross-encoder re-ranking.

        :param query: User query (any language)
        :param limit: Final number of results to return
        :param document_id: Filter to a specific document (optional).
                            When supplied together with owner_id both conditions
                            must match (AND logic) so a user can only search
                            within their own document.
        :param owner_id: UUID of the calling user.  When provided, every Qdrant
                         query is scoped to points owned by that user, preventing
                         cross-user data leakage.  Should always be set for
                         authenticated endpoints — omit only for internal/admin
                         tooling that intentionally operates across all users.
        :param fetch_k: Candidates fetched per leg (default ``settings.rag_fetch_k``)
        :param alpha: Weight of the dense leg in the fusion, 0..1
                      (default ``settings.rag_hybrid_alpha``; 1.0 = dense only,
                      0.0 = BM25 only)
        :param expansion_terms: Synonyms / canonical names from entity linking (e.g. the active
                      ingredient of a brand-name drug). Added to the dense, BM25 and re-ranker
                      query text; the caller keeps the original query for caching/prompting.
        :return: Result dicts sorted by relevance; ``score`` is in [0, 1].
        """
        if not query or not query.strip():
            return []
        query = expanded_query_text(query, expansion_terms)
        limit = max(1, limit)
        fetch_k = max(limit, fetch_k if fetch_k is not None else settings.rag_fetch_k)
        alpha = min(1.0, max(0.0, settings.rag_hybrid_alpha if alpha is None else alpha))

        try:
            from qdrant_client.http import models as qdrant_models

            self._ensure_collection()

            # Build a Qdrant Filter that enforces ownership + optional doc scope.
            # Using "must" (AND) so ALL conditions must match simultaneously.
            must_conditions = []
            if owner_id:
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="owner_id",
                        match=qdrant_models.MatchValue(value=str(owner_id)),
                    )
                )
            if document_id:
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=str(document_id)),
                    )
                )
            query_filter = qdrant_models.Filter(must=must_conditions) if must_conditions else None

            requests = []
            if alpha > 0:
                dense_vec = embed_query(query).tolist()
                requests.append(("dense", qdrant_models.QueryRequest(
                    query=dense_vec, using=DENSE_NAME, filter=query_filter,
                    limit=fetch_k, with_payload=True,
                )))
            if alpha < 1:
                s_idx, s_val = hybrid.sparse_query_vector(
                    query, bigrams=settings.rag_sparse_bigrams
                )
                if s_idx:  # query made only of stop-words/punctuation -> no keyword leg
                    requests.append(("sparse", qdrant_models.QueryRequest(
                        query=qdrant_models.SparseVector(indices=s_idx, values=s_val),
                        using=SPARSE_NAME, filter=query_filter,
                        limit=fetch_k, with_payload=True,
                    )))
            if not requests:
                return []

            responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=[r for _, r in requests],
            )

            # --- Merge both legs into one candidate pool ---
            pool: Dict[str, Dict[str, Any]] = {}
            ranked = []
            for (leg, _), resp in zip(requests, responses):
                weight = alpha if leg == "dense" else 1.0 - alpha
                keys = []
                for hit in resp.points:
                    key = str(hit.id)
                    keys.append(key)
                    cand = pool.setdefault(key, {
                        **hit_from_payload(hit.payload),
                        "vector_score": None,
                        "bm25_score": None,
                    })
                    cand["vector_score" if leg == "dense" else "bm25_score"] = float(hit.score)
                ranked.append((weight, keys))

            if not pool:
                return []

            fused = hybrid.weighted_rrf(ranked, k=settings.rag_rrf_k)
            order = sorted(fused, key=fused.get, reverse=True)
            n_rerank = max(limit, settings.rag_rerank_top_n)
            candidates = []
            for key in order[:n_rerank]:
                cand = pool[key]
                cand["rrf_score"] = fused[key]
                candidates.append(cand)

            fusion_norm = hybrid.minmax([c["rrf_score"] for c in candidates])
            for c, f in zip(candidates, fusion_norm):
                c["score"] = f

            # --- Cross-encoder re-ranking, blended with the fusion score ---
            reranker = _get_reranker()
            if reranker is not None and len(candidates) > 1:
                try:
                    pairs = [
                        (query, _sparse_text(c["content"], c["slide_title"] or c["section_title"]))
                        for c in candidates
                    ]
                    rr = [float(s) for s in reranker.predict(pairs)]
                    rr_norm = hybrid.minmax(rr)
                    w = min(1.0, max(0.0, settings.rag_rerank_weight))
                    for c, raw, rn in zip(candidates, rr, rr_norm):
                        c["rerank_score"] = raw
                        c["score"] = w * rn + (1.0 - w) * c["score"]
                except Exception as exc:
                    logger.warning(f"Re-ranking failed, using fusion order: {exc}")

            candidates.sort(key=lambda c: c["score"], reverse=True)
            return hybrid.drop_redundant_summaries(candidates)[:limit]

        except CollectionSchemaError:
            raise
        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}")
            return []

    # Legal: pull the sibling clauses of every retrieved article
    def expand_legal_articles(
        self,
        hits: List[Dict[str, Any]],
        owner_id: Optional[str] = None,
        max_extra: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return ``hits`` where each retrieved legal article is completed with its other clauses.

        A clause alone often lacks the condition / exception stated in a sibling clause
        ("... except as provided in clause 3"). Chunks of one article are grouped together in
        clause order at the position of the article's best hit; non-legal hits are untouched.
        At most ``max_extra`` chunks are added (``RAG_LEGAL_EXPAND_MAX_CHUNKS``). Fail-open.
        """
        budget = settings.rag_legal_expand_max_chunks if max_extra is None else max_extra
        if budget <= 0 or not any(h.get("article_key") for h in hits):
            return hits
        try:
            from qdrant_client.http import models as qdrant_models

            self._ensure_collection()
            seen = {(h.get("document_id"), h.get("chunk_index")) for h in hits}
            groups: Dict[tuple, List[Dict[str, Any]]] = {}
            out: List[Any] = []
            for hit in hits:
                akey = hit.get("article_key")
                if not akey:
                    out.append(hit)
                    continue
                gkey = (hit.get("document_id"), akey)
                if gkey in groups:
                    groups[gkey].append(hit)
                    continue
                groups[gkey] = [hit]
                out.append(gkey)  # placeholder: the article group goes here
                if budget <= 0:
                    continue
                must = [
                    qdrant_models.FieldCondition(key="document_id",
                                                 match=qdrant_models.MatchValue(value=str(gkey[0]))),
                    qdrant_models.FieldCondition(key="article_key",
                                                 match=qdrant_models.MatchValue(value=str(akey))),
                ]
                if owner_id:
                    must.append(qdrant_models.FieldCondition(
                        key="owner_id", match=qdrant_models.MatchValue(value=str(owner_id))))
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_models.Filter(must=must),
                    limit=64, with_payload=True, with_vectors=False,
                )
                siblings = sorted(
                    (p.payload for p in points if p.payload and not p.payload.get("is_summary")),
                    key=lambda p: p.get("chunk_index", 0),
                )
                for payload in siblings:
                    ident = (payload.get("document_id"), payload.get("chunk_index"))
                    if ident in seen or budget <= 0:
                        continue
                    seen.add(ident)
                    budget -= 1
                    sibling = hit_from_payload(payload, score=float(hit.get("score", 0.0)))
                    sibling["expanded"] = True
                    groups[gkey].append(sibling)
            result: List[Dict[str, Any]] = []
            for item in out:
                if isinstance(item, tuple):
                    result.extend(sorted(groups[item], key=lambda h: h.get("chunk_index", 0)))
                else:
                    result.append(item)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Legal article expansion failed, using plain hits: {exc}")
            return hits


# Module-level singleton
vector_store = VectorStoreManager()
