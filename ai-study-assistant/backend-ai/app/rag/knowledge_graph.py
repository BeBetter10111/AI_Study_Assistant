"""GraphRAG: extract a per-document knowledge graph, persist it, and use it at query time.

Students in social sciences / biomed / philosophy often need *relations* between concepts
(``X causes Y``, ``A is a kind of B``), not only nearest passages.  After chunks are embedded
we extract (subject, relation, object) triples — heuristically by default, optionally with one
structured LLM call per batch — store them in Postgres (``kg_nodes`` / ``kg_edges``; Neo4j is
optional), then:

* ``facts_for_query``  — 1-hop (or ``kg_hops``) neighbourhood as prompt context;
* ``mindmap_for_document`` — nodes + edges for a study mindmap UI.

Fail-open everywhere: extraction / lookup errors never fail document processing or chat.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.models import KGEdge, KGNode
from app.rag.hybrid import fold_diacritics

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the this to was were
    will with what which who how why when where do does did not no
    và là của có các một những được trong cho với này để không khi thì mà như từ cũng đã sẽ bị
    tại về theo thế nào gì sao ra vào lại rất nên cần hay hoặc nếu vì do bởi
    """.split()
)

# "X is a/an Y", "X causes Y", "X thuộc về Y", "X gây ra Y", ...
_REL_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"(?P<s>[A-ZÀ-Ỹ][\wÀ-ỹ\- ]{1,60}?)\s+(?:is|are|was|were)\s+(?:a|an|the)\s+"
        r"(?P<o>[A-Za-zÀ-ỹ][\wÀ-ỹ\- ]{1,60})",
        re.UNICODE,
    ), "is_a"),
    (re.compile(
        r"(?P<s>[A-ZÀ-Ỹ][\wÀ-ỹ\- ]{1,60}?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+"
        r"(?P<o>[A-Za-zÀ-ỹ][\wÀ-ỹ\- ]{1,60})",
        re.IGNORECASE | re.UNICODE,
    ), "causes"),
    (re.compile(
        r"(?P<s>[A-ZÀ-Ỹ][\wÀ-ỹ\- ]{1,60}?)\s+(?:includes?|contains?|consists?\s+of)\s+"
        r"(?P<o>[A-Za-zÀ-ỹ][\wÀ-ỹ\- ]{1,60})",
        re.IGNORECASE | re.UNICODE,
    ), "includes"),
    (re.compile(
        r"(?P<s>[\wÀ-ỹ][\wÀ-ỹ\- ]{1,60}?)\s+(?:là một|là một loại|là)\s+"
        r"(?P<o>[\wÀ-ỹ][\wÀ-ỹ\- ]{1,60})",
        re.IGNORECASE | re.UNICODE,
    ), "is_a"),
    (re.compile(
        r"(?P<s>[\wÀ-ỹ][\wÀ-ỹ\- ]{1,60}?)\s+(?:gây ra|dẫn đến|làm cho)\s+"
        r"(?P<o>[\wÀ-ỹ][\wÀ-ỹ\- ]{1,60})",
        re.IGNORECASE | re.UNICODE,
    ), "causes"),
    (re.compile(
        r"(?P<s>[\wÀ-ỹ][\wÀ-ỹ\- ]{1,60}?)\s+(?:thuộc về|nằm trong|liên quan đến|liên quan tới)\s+"
        r"(?P<o>[\wÀ-ỹ][\wÀ-ỹ\- ]{1,60})",
        re.IGNORECASE | re.UNICODE,
    ), "related_to"),
]

_ALLOWED_RELATIONS = frozenset({
    "is_a", "causes", "includes", "related_to", "part_of", "defines", "contrasts_with", "uses",
})


def _norm_name(name: str) -> str:
    return " ".join(_WORD_RE.findall(fold_diacritics(name or "").lower()))


def _clean_entity(raw: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", " ", (raw or "").strip(" .,:;\"'()[]{}"))
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] or text[:max_len]
    return text.strip()


def _is_good_entity(name: str) -> bool:
    tokens = _WORD_RE.findall(name)
    if not tokens or len(tokens) > 8:
        return False
    if all(t.lower() in _STOP for t in tokens):
        return False
    if len(name) < 2 or len(name) > 120:
        return False
    return True


@dataclass
class Triple:
    source: str
    relation: str
    target: str
    source_type: str = "concept"
    target_type: str = "concept"
    chunk_index: Optional[int] = None

    def normalised(self) -> Optional["Triple"]:
        s, t = _clean_entity(self.source), _clean_entity(self.target)
        rel = (self.relation or "related_to").strip().lower().replace(" ", "_")
        if rel not in _ALLOWED_RELATIONS:
            rel = "related_to"
        if not _is_good_entity(s) or not _is_good_entity(t):
            return None
        if _norm_name(s) == _norm_name(t):
            return None
        return Triple(s, rel, t, self.source_type, self.target_type, self.chunk_index)


@dataclass
class ExtractedGraph:
    triples: List[Triple] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.triples)


class _LLMTriple(BaseModel):
    source: str
    relation: str = "related_to"
    target: str
    source_type: str = "concept"
    target_type: str = "concept"


class _LLMGraphBatch(BaseModel):
    triples: List[_LLMTriple] = Field(default_factory=list)


# Sync DB (Celery worker + threadpool from API)
_sync_engine = None


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(settings.database_url.replace("+asyncpg", ""), pool_pre_ping=True)
    return _sync_engine


# Extraction
def extract_heuristic(chunks: Sequence[Any]) -> ExtractedGraph:
    """Rule-based triples from chunk text (no LLM). ``chunks`` are ``ChunkData`` or str."""
    triples: List[Triple] = []
    seen: Set[Tuple[str, str, str]] = set()
    for idx, chunk in enumerate(chunks):
        if getattr(chunk, "extra", {}).get("is_summary"):
            continue
        text = getattr(chunk, "content", None) or str(chunk)
        for line in text.splitlines():
            line = line.strip()
            if len(line) < 8 or len(line) > 400:
                continue
            for pattern, relation in _REL_PATTERNS:
                m = pattern.search(line)
                if not m:
                    continue
                tri = Triple(
                    m.group("s"), relation, m.group("o"), chunk_index=idx,
                ).normalised()
                if tri is None:
                    continue
                key = (_norm_name(tri.source), tri.relation, _norm_name(tri.target))
                if key in seen:
                    continue
                seen.add(key)
                triples.append(tri)
    return ExtractedGraph(triples=triples)


def extract_with_llm(chunks: Sequence[Any]) -> ExtractedGraph:
    """Structured LLM extraction over the first ``kg_max_chunks_per_doc`` content chunks."""
    from app.core.llm_engine import llm_engine

    content_chunks = [
        (i, getattr(c, "content", str(c)))
        for i, c in enumerate(chunks)
        if not getattr(c, "extra", {}).get("is_summary")
    ][: max(1, settings.kg_max_chunks_per_doc)]

    batch = max(1, settings.kg_batch_size)
    triples: List[Triple] = []
    seen: Set[Tuple[str, str, str]] = set()

    system = (
        "Extract concept relations for a study knowledge graph. "
        "Return only important textbook concepts and clear relations. "
        f"Allowed relations: {', '.join(sorted(_ALLOWED_RELATIONS))}. "
        "Prefer nouns/noun phrases; skip page numbers and generic words."
    )

    for start in range(0, len(content_chunks), batch):
        group = content_chunks[start:start + batch]
        numbered = "\n\n".join(f"[chunk {i}]\n{text[:1200]}" for i, text in group)
        prompt = (
            f"{system}\n\nPassages:\n{numbered}\n\n"
            "Extract up to 12 triples as JSON matching the schema."
        )
        try:
            data = llm_engine.generate_structured_output(prompt, _LLMGraphBatch, temperature=0.0)
            for raw in data.get("triples") or []:
                # Map triples back to the first chunk of the batch when unsure.
                tri = Triple(
                    raw.get("source", ""),
                    raw.get("relation", "related_to"),
                    raw.get("target", ""),
                    raw.get("source_type") or "concept",
                    raw.get("target_type") or "concept",
                    chunk_index=group[0][0],
                ).normalised()
                if tri is None:
                    continue
                key = (_norm_name(tri.source), tri.relation, _norm_name(tri.target))
                if key in seen:
                    continue
                seen.add(key)
                triples.append(tri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KG LLM extraction batch failed (fail-open): %s", exc)
    return ExtractedGraph(triples=triples)


def extract_graph(chunks: Sequence[Any]) -> ExtractedGraph:
    heuristic = extract_heuristic(chunks)
    if not settings.kg_use_llm:
        return heuristic
    llm_graph = extract_with_llm(chunks)
    if not llm_graph:
        return heuristic
    # Merge: LLM first, then heuristic fillers.
    seen = {(_norm_name(t.source), t.relation, _norm_name(t.target)) for t in llm_graph.triples}
    merged = list(llm_graph.triples)
    for t in heuristic.triples:
        key = (_norm_name(t.source), t.relation, _norm_name(t.target))
        if key not in seen:
            seen.add(key)
            merged.append(t)
    return ExtractedGraph(triples=merged)


# Persist (Postgres)
def clear_document_graph(session: Session, document_id: str, owner_id: str) -> None:
    # Edges first (FK), then nodes.
    session.execute(
        delete(KGEdge).where(KGEdge.document_id == document_id, KGEdge.owner_id == owner_id)
    )
    session.execute(
        delete(KGNode).where(KGNode.document_id == document_id, KGNode.owner_id == owner_id)
    )


def persist_graph(
    graph: ExtractedGraph,
    document_id: str,
    owner_id: str,
    session: Optional[Session] = None,
) -> Dict[str, int]:
    """Replace the graph for one document. Returns ``{nodes, edges}`` counts."""
    if not owner_id or not document_id:
        return {"nodes": 0, "edges": 0}

    owns_session = session is None
    if owns_session:
        session = Session(_get_sync_engine())
    assert session is not None

    try:
        clear_document_graph(session, document_id, owner_id)
        if not graph.triples:
            session.commit()
            return {"nodes": 0, "edges": 0}

        node_ids: Dict[str, uuid.UUID] = {}
        chunk_map: Dict[str, Set[int]] = {}
        type_map: Dict[str, str] = {}

        def _ensure(name: str, ntype: str, chunk_index: Optional[int]) -> str:
            key = _norm_name(name)
            if key not in node_ids:
                node_ids[key] = uuid.uuid4()
                chunk_map[key] = set()
                type_map[key] = ntype or "concept"
            if chunk_index is not None:
                chunk_map[key].add(int(chunk_index))
            # Prefer the first display name we saw (capitalisation from source text).
            return key

        display: Dict[str, str] = {}
        for tri in graph.triples:
            sk = _ensure(tri.source, tri.source_type, tri.chunk_index)
            tk = _ensure(tri.target, tri.target_type, tri.chunk_index)
            display.setdefault(sk, tri.source)
            display.setdefault(tk, tri.target)

        for key, nid in node_ids.items():
            session.add(KGNode(
                id=nid,
                owner_id=uuid.UUID(str(owner_id)),
                document_id=uuid.UUID(str(document_id)),
                name=display[key][:255],
                norm_name=key[:255],
                type=(type_map.get(key) or "concept")[:64],
                chunk_indices=json.dumps(sorted(chunk_map.get(key) or [])),
            ))
        session.flush()

        edge_seen: Set[Tuple[str, str, str]] = set()
        edge_count = 0
        for tri in graph.triples:
            sk, tk = _norm_name(tri.source), _norm_name(tri.target)
            ekey = (sk, tri.relation, tk)
            if ekey in edge_seen or sk not in node_ids or tk not in node_ids:
                continue
            edge_seen.add(ekey)
            session.add(KGEdge(
                id=uuid.uuid4(),
                owner_id=uuid.UUID(str(owner_id)),
                document_id=uuid.UUID(str(document_id)),
                source_id=node_ids[sk],
                target_id=node_ids[tk],
                relation=tri.relation[:128],
                weight=1.0,
                chunk_index=tri.chunk_index,
            ))
            edge_count += 1

        session.commit()
        counts = {"nodes": len(node_ids), "edges": edge_count}
        if settings.kg_backend == "neo4j" and settings.neo4j_uri:
            try:
                _mirror_to_neo4j(document_id, owner_id, graph)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neo4j mirror failed (Postgres kept): %s", exc)
        return counts
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def build_and_store(chunks: Sequence[Any], document_id: str, owner_id: str) -> Dict[str, int]:
    """Full pipeline for the Celery worker. No-op when ``kg_enabled`` is false."""
    if not settings.kg_enabled:
        return {"nodes": 0, "edges": 0, "skipped": 1}
    if not owner_id:
        logger.warning("KG build skipped: missing owner_id for document %s", document_id)
        return {"nodes": 0, "edges": 0, "skipped": 1}
    try:
        graph = extract_graph(chunks)
        return persist_graph(graph, document_id, owner_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("KG build failed for document %s: %s", document_id, exc)
        return {"nodes": 0, "edges": 0, "error": 1}


# Optional Neo4j mirror
def _mirror_to_neo4j(document_id: str, owner_id: str, graph: ExtractedGraph) -> None:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError as exc:
        raise RuntimeError("neo4j driver not installed") from exc

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password or ""),
    )
    try:
        with driver.session() as neo:
            neo.run(
                "MATCH (n:Concept {document_id: $d}) DETACH DELETE n",
                d=str(document_id),
            )
            for tri in graph.triples:
                neo.run(
                    """
                    MERGE (s:Concept {document_id: $d, owner_id: $o, norm: $sn})
                      ON CREATE SET s.name = $sname
                    MERGE (t:Concept {document_id: $d, owner_id: $o, norm: $tn})
                      ON CREATE SET t.name = $tname
                    MERGE (s)-[r:REL {type: $rel}]->(t)
                    """,
                    d=str(document_id), o=str(owner_id),
                    sn=_norm_name(tri.source), sname=tri.source,
                    tn=_norm_name(tri.target), tname=tri.target,
                    rel=tri.relation,
                )
    finally:
        driver.close()


# Query-time facts + mindmap
def _match_seed_nodes(
    session: Session,
    owner_id: str,
    query: str,
    document_id: Optional[str] = None,
    limit: int = 8,
) -> List[KGNode]:
    tokens = [t for t in _WORD_RE.findall(fold_diacritics(query.lower())) if t not in _STOP and len(t) > 1]
    if not tokens:
        return []
    q = select(KGNode).where(KGNode.owner_id == owner_id)
    if document_id:
        q = q.where(KGNode.document_id == document_id)
    nodes = list(session.scalars(q).all())
    scored: List[Tuple[int, KGNode]] = []
    for node in nodes:
        name = node.norm_name or ""
        score = sum(1 for t in tokens if t in name)
        if score:
            scored.append((score, node))
    scored.sort(key=lambda x: (-x[0], x[1].name))
    return [n for _, n in scored[:limit]]


def facts_for_query(
    query: str,
    owner_id: str,
    document_id: Optional[str] = None,
    hops: Optional[int] = None,
    max_facts: Optional[int] = None,
) -> List[str]:
    """Return human-readable facts ``A --relation--> B`` near query concepts."""
    if not settings.kg_enabled or not owner_id or not (query or "").strip():
        return []
    hops = settings.kg_hops if hops is None else hops
    max_facts = settings.kg_max_facts if max_facts is None else max_facts
    try:
        with Session(_get_sync_engine()) as session:
            seeds = _match_seed_nodes(session, owner_id, query, document_id)
            if not seeds:
                return []
            frontier: Set[uuid.UUID] = {n.id for n in seeds}
            known = {n.id: n for n in seeds}
            facts: List[str] = []
            seen_edge: Set[uuid.UUID] = set()

            for _ in range(max(0, hops)):
                if not frontier or len(facts) >= max_facts:
                    break
                edge_q = select(KGEdge).where(
                    KGEdge.owner_id == owner_id,
                    (KGEdge.source_id.in_(frontier)) | (KGEdge.target_id.in_(frontier)),
                )
                if document_id:
                    edge_q = edge_q.where(KGEdge.document_id == document_id)
                edges = list(session.scalars(edge_q).all())
                next_frontier: Set[uuid.UUID] = set()
                needed: Set[uuid.UUID] = set()
                for e in edges:
                    if e.id in seen_edge:
                        continue
                    needed.add(e.source_id)
                    needed.add(e.target_id)
                missing = [i for i in needed if i not in known]
                if missing:
                    for n in session.scalars(select(KGNode).where(KGNode.id.in_(missing))).all():
                        known[n.id] = n
                for e in edges:
                    if e.id in seen_edge:
                        continue
                    seen_edge.add(e.id)
                    src, tgt = known.get(e.source_id), known.get(e.target_id)
                    if not src or not tgt:
                        continue
                    facts.append(f"{src.name} --{e.relation}--> {tgt.name}")
                    if e.source_id not in frontier:
                        next_frontier.add(e.source_id)
                    if e.target_id not in frontier:
                        next_frontier.add(e.target_id)
                    if len(facts) >= max_facts:
                        break
                frontier = next_frontier - {n.id for n in seeds}
            return facts[:max_facts]
    except Exception as exc:  # noqa: BLE001
        logger.warning("KG facts_for_query failed: %s", exc)
        return []


def mindmap_for_document(document_id: str, owner_id: str) -> Dict[str, Any]:
    """Serialise the document graph for a mindmap UI: ``{nodes, edges, stats}``."""
    empty = {"nodes": [], "edges": [], "stats": {"nodes": 0, "edges": 0}}
    if not owner_id or not document_id:
        return empty
    try:
        with Session(_get_sync_engine()) as session:
            nodes = list(session.scalars(
                select(KGNode).where(
                    KGNode.document_id == document_id, KGNode.owner_id == owner_id,
                )
            ).all())
            edges = list(session.scalars(
                select(KGEdge).where(
                    KGEdge.document_id == document_id, KGEdge.owner_id == owner_id,
                )
            ).all())
            node_payload = [
                {
                    "id": str(n.id),
                    "name": n.name,
                    "type": n.type or "concept",
                    "chunk_indices": json.loads(n.chunk_indices or "[]"),
                }
                for n in nodes
            ]
            edge_payload = [
                {
                    "id": str(e.id),
                    "source": str(e.source_id),
                    "target": str(e.target_id),
                    "relation": e.relation,
                    "weight": e.weight,
                    "chunk_index": e.chunk_index,
                }
                for e in edges
            ]
            return {
                "nodes": node_payload,
                "edges": edge_payload,
                "stats": {"nodes": len(node_payload), "edges": len(edge_payload)},
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("mindmap_for_document failed: %s", exc)
        return empty


def format_facts_block(facts: Iterable[str]) -> str:
    lines = [f for f in facts if f]
    if not lines:
        return ""
    return "Knowledge-graph relations:\n" + "\n".join(f"- {line}" for line in lines)
