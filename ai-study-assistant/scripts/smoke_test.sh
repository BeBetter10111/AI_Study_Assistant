#!/usr/bin/env bash
# Smoke test tối thiểu sau khi deploy — "hít khói xem có cháy không" trước
# khi tin bản mới. Dùng cả trong CD pipeline (deploy-staging/production) lẫn
# chạy tay sau khi `docker compose up -d` ở local.
#
# Cách dùng: BASE_URL=http://localhost:8000 ./scripts/smoke_test.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_DELAY_S="${RETRY_DELAY_S:-3}"

log() { echo "[smoke_test] $*"; }

# 1) Chờ /health trả 200 — có retry vì container vừa "up" có thể cần vài giây
#    để backend kết nối xong Postgres/Qdrant/MinIO (xem app/main.py).
log "Chờ $BASE_URL/health sẵn sàng..."
attempt=1
until curl --silent --fail --max-time 5 "$BASE_URL/health" > /tmp/health_response.json; do
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    log "THẤT BẠI: /health không trả 200 sau $MAX_RETRIES lần thử."
    exit 1
  fi
  log "Lần $attempt/$MAX_RETRIES: chưa sẵn sàng, thử lại sau ${RETRY_DELAY_S}s..."
  attempt=$((attempt + 1))
  sleep "$RETRY_DELAY_S"
done
log "OK: /health = $(cat /tmp/health_response.json)"

# 2) Gọi 1-2 API quan trọng — không cần thành công theo nghĩa "trả dữ liệu
#    đúng", chỉ cần server không sập/504/502 với payload hợp lệ tối thiểu.
log "Kiểm tra /api/chat/query phản hồi (không cần có ngữ cảnh)..."
chat_status=$(curl --silent --output /dev/null --write-out "%{http_code}" --max-time 10 \
  -X POST "$BASE_URL/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "smoke test ping"}')

if [ "$chat_status" != "200" ]; then
  log "THẤT BẠI: /api/chat/query trả HTTP $chat_status (mong đợi 200)."
  exit 1
fi
log "OK: /api/chat/query trả HTTP 200."

log "Kiểm tra /api/auth/login trả lỗi hợp lệ cho tài khoản không tồn tại (không phải 500)..."
auth_status=$(curl --silent --output /dev/null --write-out "%{http_code}" --max-time 10 \
  -X POST "$BASE_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "smoke-test@example.com", "password": "does-not-matter"}')

if [ "$auth_status" != "401" ]; then
  log "THẤT BẠI: /api/auth/login trả HTTP $auth_status (mong đợi 401, không phải 500)."
  exit 1
fi
log "OK: /api/auth/login trả HTTP 401 như mong đợi (không crash)."

log "SMOKE TEST PASS — bản deploy có vẻ sống và phản hồi đúng."
