CREATE TABLE IF NOT EXISTS users (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(150) UNIQUE NOT NULL,
    full_name     VARCHAR(255),
    email         VARCHAR(255) UNIQUE NOT NULL,
    phone         VARCHAR(30)  UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role          VARCHAR(50)  NOT NULL DEFAULT 'Staff',
    created_at    TIMESTAMPTZ  DEFAULT now(),
    password_changed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      VARCHAR(64) NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    verified_at     TIMESTAMPTZ,
    used            BOOLEAN     NOT NULL DEFAULT false,
    failed_attempts INT         NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prt_token_hash ON password_reset_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_prt_user_id    ON password_reset_tokens(user_id);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES users(id),
    filename VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128),
    status VARCHAR(32) DEFAULT 'pending',
    storage_path VARCHAR(512),            
    size_bytes BIGINT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    document_id UUID REFERENCES documents(id),
    quiz_id UUID,
    is_correct BOOLEAN,
    response_time_s FLOAT,
    hints_used INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_rl_states (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    difficulty_index INT NOT NULL DEFAULT 0,
    reward_history_json TEXT NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL DEFAULT '',
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT,
    intent VARCHAR(32),
    cached BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conv ON chat_messages(conversation_id, created_at);

-- Knowledge graph (GraphRAG / mindmap), see scripts/migrations/004_knowledge_graph.sql
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

-- Exam-prep mode and lecturer question bank, see scripts/migrations/005_exam_lecturer.sql
CREATE TABLE IF NOT EXISTS exams (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
        title VARCHAR(200) NOT NULL DEFAULT '',
        syllabus TEXT NOT NULL DEFAULT '',
        num_questions INT NOT NULL DEFAULT 0,
        duration_min INT,
        created_at TIMESTAMPTZ DEFAULT now()
    );

CREATE TABLE IF NOT EXISTS exam_questions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        exam_id UUID NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
        idx INT NOT NULL,
        topic VARCHAR(255) NOT NULL DEFAULT '',
        question TEXT NOT NULL,
        options_json TEXT NOT NULL DEFAULT '[]',
        correct_index INT NOT NULL,
        explanation TEXT NOT NULL DEFAULT ''
    );

CREATE TABLE IF NOT EXISTS exam_attempts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        exam_id UUID NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
        owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        answers_json TEXT NOT NULL DEFAULT '[]',
        correct INT NOT NULL DEFAULT 0,
        total INT NOT NULL DEFAULT 0,
        score FLOAT NOT NULL DEFAULT 0,
        per_topic_json TEXT NOT NULL DEFAULT '{}',
        duration_s FLOAT,
        created_at TIMESTAMPTZ DEFAULT now()
    );

CREATE TABLE IF NOT EXISTS question_bank_items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
        topic VARCHAR(255) NOT NULL DEFAULT '',
        qtype VARCHAR(16) NOT NULL DEFAULT 'mcq',
        difficulty VARCHAR(8) NOT NULL DEFAULT 'medium',
        question TEXT NOT NULL,
        options_json TEXT,
        correct_index INT,
        answer_key TEXT NOT NULL DEFAULT '',
        rubric_json TEXT,
        explanation TEXT NOT NULL DEFAULT '',
        source_ref VARCHAR(255),
        status VARCHAR(16) NOT NULL DEFAULT 'draft',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );

CREATE TABLE IF NOT EXISTS exam_papers (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title VARCHAR(200) NOT NULL DEFAULT '',
        snapshot_json TEXT NOT NULL DEFAULT '{}',
        total_points FLOAT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT now()
    );
CREATE INDEX IF NOT EXISTS idx_exams_owner ON exams(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exam_questions_exam ON exam_questions(exam_id, idx);
CREATE INDEX IF NOT EXISTS idx_exam_attempts_owner ON exam_attempts(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exam_attempts_exam ON exam_attempts(exam_id);
CREATE INDEX IF NOT EXISTS idx_bank_owner ON question_bank_items(owner_id, topic);
CREATE INDEX IF NOT EXISTS idx_papers_owner ON exam_papers(owner_id, created_at DESC);
