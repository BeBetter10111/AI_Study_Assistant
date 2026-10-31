-- Chạy MỘT LẦN trên database đã tồn tại (init_db.sql chỉ chạy khi volume Postgres còn trống).
--   docker compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB < backend-ai/scripts/migrations/001_password_reset.sql

ALTER TABLE users ADD COLUMN IF NOT EXISTS password_changed_at TIMESTAMPTZ;

-- Bảng cũ chưa được code nào sử dụng (cột token VARCHAR(36) không đủ chỗ cho hash) -> tạo lại.
DROP TABLE IF EXISTS password_reset_tokens;
CREATE TABLE password_reset_tokens (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      VARCHAR(64) NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    verified_at     TIMESTAMPTZ,
    used            BOOLEAN     NOT NULL DEFAULT false,
    failed_attempts INT         NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_prt_token_hash ON password_reset_tokens(token_hash);
CREATE INDEX idx_prt_user_id    ON password_reset_tokens(user_id);
