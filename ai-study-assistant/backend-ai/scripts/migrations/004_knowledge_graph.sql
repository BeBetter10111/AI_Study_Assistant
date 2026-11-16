-- Knowledge graph (GraphRAG / mindmap) extracted per document. Idempotent; the backend also creates these
-- tables at startup (app/core/schema.py), so this file is needed only when AUTO_MIGRATE_SCHEMA=false.
CREATE TABLE IF NOT EXISTS kg_nodes (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id      UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id   UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    norm_name     VARCHAR(255) NOT NULL,
    type          VARCHAR(64),
    chunk_indices TEXT         NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id    UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_id   UUID         NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    target_id   UUID         NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    relation    VARCHAR(128) NOT NULL DEFAULT 'related_to',
    weight      FLOAT        NOT NULL DEFAULT 1.0,
    chunk_index INT
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_owner_doc ON kg_nodes(owner_id, document_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_norm      ON kg_nodes(owner_id, norm_name);
CREATE INDEX IF NOT EXISTS idx_kg_edges_owner_doc ON kg_edges(owner_id, document_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source    ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target    ON kg_edges(target_id);
