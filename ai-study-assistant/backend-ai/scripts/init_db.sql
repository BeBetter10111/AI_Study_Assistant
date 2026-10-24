CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES users(id),
    filename VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128),
    status VARCHAR(32) DEFAULT 'pending',
    storage_path VARCHAR(512),            
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
