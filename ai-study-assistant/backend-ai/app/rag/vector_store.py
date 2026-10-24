"""Vector store — multilingual embeddings, true hybrid (dense + BM25) search, re-ranking.

Pipeline of ``search_similar_chunks``
-------------------------------------
1. **Dense leg**  — multilingual mpnet embedding, cosine search in Qdrant.
2. **Sparse leg** — BM25 term weights stored as a Qdrant sparse vector (IDF is
   computed by Qdrant over the whole collection), searched *independently*
   so exact keywords / course codes / function names are retrieved even when
   the dense model misses them.  Both legs run in one ``query_batch_points``
   round-trip and share the same ``document_id`` filter.
3. **Fusion**     — weighted Reciprocal Rank Fusion (``alpha`` = dense weight).
4. **Re-ranking** — multilingual cross-encoder on the top candidates; its score
   is *blended* with the fusion score (not a hard override).
5. **Clean-up**   — drop summary chunks that duplicate a returned detail chunk.

Schema (breaking change)
------------------------
The collection now uses named vectors ``dense`` + sparse ``bm25``.  A
collection created by the previous version must be deleted and its documents
re-uploaded (``scripts/bulk_upload.py``).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

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


class CollectionSchemaError(RuntimeError):
    """The Qdrant collection does not match what this code version expects."""


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading multilingual embedding model …")
        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        logger.info("Embedding model ready.")
    return _embedding_model


def _get_reranker():
    """Return the cross-encoder, or None if disabled / unavailable."""
    global _reranker, _reranker_failed
    if not settings.rag_rerank_enabled or _reranker_failed:
        return None
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder re-ranker {settings.rag_reranker_model} …")
            _reranker = CrossEncoder(settings.rag_reranker_model, max_length=512)
            logger.info("Re-ranker ready.")
        except Exception as exc:
            logger.warning(f"Could not load re-ranker — falling back to fusion-only ranking: {exc}")
            _reranker_failed = True
            return None
    return _reranker


def _sparse_text(content: str, title: Optional[str]) -> str:
    """Titles carry strong keyword signal, so they are indexed with the body."""
    return f"{title}\n{content}" if title else content


def _title_of(payload: Dict[str, Any]) -> Optional[str]:
    return payload.get("slide_title") or payload.get("section_title")


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
        except Exception as e:
            logger.error(f"Error initialising Qdrant collection: {e}")
            raise

    # Upsert — accepts ChunkData objects with rich metadata
    def upsert_chunks(
        self,
        chunks: List[ChunkData],
        document_id: str,
        start_chunk_idx: int = 0,
    ) -> bool:
        """Embed chunks (dense + BM25 sparse) and upsert into Qdrant.

        :param chunks: List of ``ChunkData`` from ingestion.py
        :param document_id: Parent document UUID
        :param start_chunk_idx: Offset for chunk_index (for incremental ingestion)
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
                # Persist structural metadata — only include keys that have values
                if chunk.page_number is not None:
                    payload["page_number"] = chunk.page_number
                if chunk.slide_number is not None:
                    payload["slide_number"] = chunk.slide_number
                if chunk.slide_title:
                    payload["slide_title"] = chunk.slide_title
                if chunk.section_title:
                    payload["section_title"] = chunk.section_title

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
    def search_similar_chunks(
        self,
        query: str,
        limit: int = 4,
        document_id: Optional[str] = None,
        fetch_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid dense + BM25 retrieval, RRF fusion, cross-encoder re-ranking.

        :param query: User query (any language)
        :param limit: Final number of results to return
        :param document_id: Filter to a specific document (optional)
        :param fetch_k: Candidates fetched per leg (default ``settings.rag_fetch_k``)
        :param alpha: Weight of the dense leg in the fusion, 0..1
                      (default ``settings.rag_hybrid_alpha``; 1.0 = dense only,
                      0.0 = BM25 only)
        :return: Result dicts sorted by relevance; ``score`` is in [0, 1].
        """
        if not query or not query.strip():
            return []
        limit = max(1, limit)
        fetch_k = max(limit, fetch_k if fetch_k is not None else settings.rag_fetch_k)
        alpha = min(1.0, max(0.0, settings.rag_hybrid_alpha if alpha is None else alpha))

        try:
            from qdrant_client.http import models as qdrant_models

            self._ensure_collection()

            query_filter = None
            if document_id:
                query_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=str(document_id)),
                        )
                    ]
                )

            requests = []
            if alpha > 0:
                dense_vec = _get_embedding_model().encode(query).tolist()
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
                        "content": hit.payload.get("content", ""),
                        "document_id": hit.payload.get("document_id", ""),
                        "chunk_index": hit.payload.get("chunk_index", 0),
                        "page_number": hit.payload.get("page_number"),
                        "slide_number": hit.payload.get("slide_number"),
                        "slide_title": hit.payload.get("slide_title"),
                        "section_title": hit.payload.get("section_title"),
                        "is_summary": hit.payload.get("is_summary", False),
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


# Module-level singleton
vector_store = VectorStoreManager()
