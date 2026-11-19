-- Exam-prep mode (exams, questions, attempts) and lecturer mode (question bank, papers). Idempotent; the backend
-- also creates these tables at startup (app/core/schema.py), so this file is needed only when AUTO_MIGRATE_SCHEMA=false.
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
