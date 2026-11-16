-- Original file size used to calculate each user's storage quota. Idempotent; the backend also adds this column at startup
-- (app/core/schema.py), so this file is needed only when AUTO_MIGRATE_SCHEMA=false.
-- Older documents have size_bytes = NULL (counted as 0 bytes when calculating the quota total).
ALTER TABLE documents ADD COLUMN IF NOT EXISTS size_bytes BIGINT;
