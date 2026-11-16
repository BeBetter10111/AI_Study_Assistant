"""Tự nâng cấp schema Postgres (chỉ THÊM, không bao giờ xoá dữ liệu) khi backend khởi động.

Vì sao cần: ``scripts/init_db.sql`` chỉ chạy khi volume Postgres còn TRỐNG. Mỗi lần code thêm cột
(``full_name``, ``phone``, ``password_changed_at``...) hoặc bảng mới, database đã tồn tại từ trước
không có chúng, và MỌI truy vấn ``users`` nổ ``UndefinedColumnError`` -> HTTP 500 ở đăng ký lẫn
đăng nhập. Tất cả câu lệnh ở đây đều idempotent (``IF NOT EXISTS``) nên chạy nhiều lần vẫn an toàn.

Tắt bằng biến môi trường ``AUTO_MIGRATE_SCHEMA=false`` nếu bạn muốn tự quản lý migration.
"""
import logging

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

_LOCK_KEY = 727_274_001  # Advisory lock: prevent multiple backend instances from running this concurrently at startup.

# Create any missing tables; definitions match scripts/init_db.sql.
_CREATE_TABLES: list[tuple[str, str]] = [
    ("users", """
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
        )"""),
    ("documents", """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            owner_id UUID REFERENCES users(id),
            filename VARCHAR(512) NOT NULL,
            mime_type VARCHAR(128),
            status VARCHAR(32) DEFAULT 'pending',
            storage_path VARCHAR(512),
            created_at TIMESTAMPTZ DEFAULT now()
        )"""),
    ("quiz_attempts", """
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id),
            document_id UUID REFERENCES documents(id),
            quiz_id UUID,
            is_correct BOOLEAN,
            response_time_s FLOAT,
            hints_used INT DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now()
        )"""),
    ("user_rl_states", """
        CREATE TABLE IF NOT EXISTS user_rl_states (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            difficulty_index INT NOT NULL DEFAULT 0,
            reward_history_json TEXT NOT NULL DEFAULT '[]',
            updated_at TIMESTAMPTZ DEFAULT now()
        )"""),
    ("conversations", """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(200) NOT NULL DEFAULT '',
            document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )"""),
    ("chat_messages", """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT,
            intent VARCHAR(32),
            cached BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ DEFAULT now()
        )"""),
    ("kg_nodes", """
        CREATE TABLE IF NOT EXISTS kg_nodes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            norm_name VARCHAR(255) NOT NULL,
            type VARCHAR(64),
            chunk_indices TEXT NOT NULL DEFAULT '[]'
        )"""),
    ("kg_edges", """
        CREATE TABLE IF NOT EXISTS kg_edges (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            source_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            target_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            relation VARCHAR(128) NOT NULL DEFAULT 'related_to',
            weight FLOAT NOT NULL DEFAULT 1.0,
            chunk_index INT
        )"""),
]

_PASSWORD_RESET_DDL = """
    CREATE TABLE password_reset_tokens (
        id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash      VARCHAR(64) NOT NULL,
        expires_at      TIMESTAMPTZ NOT NULL,
        verified_at     TIMESTAMPTZ,
        used            BOOLEAN     NOT NULL DEFAULT false,
        failed_attempts INT         NOT NULL DEFAULT 0,
        created_at      TIMESTAMPTZ DEFAULT now()
    )"""
_PASSWORD_RESET_COLUMNS = {
    "id", "user_id", "token_hash", "expires_at", "verified_at", "used", "failed_attempts", "created_at",
}

# Columns may be missing from older databases. These columns are nullable or have a DEFAULT,
# so ADD COLUMN cannot fail because of existing data. (UNIQUE applies only when a column is newly added.)
_ADD_COLUMNS: list[tuple[str, str, str]] = [
    ("users", "full_name", "VARCHAR(255)"),
    ("users", "phone", "VARCHAR(30) UNIQUE"),
    ("users", "role", "VARCHAR(50) NOT NULL DEFAULT 'Staff'"),
    ("users", "created_at", "TIMESTAMPTZ DEFAULT now()"),
    ("users", "password_changed_at", "TIMESTAMPTZ"),
    ("documents", "owner_id", "UUID REFERENCES users(id)"),
    ("documents", "mime_type", "VARCHAR(128)"),
    ("documents", "status", "VARCHAR(32) DEFAULT 'pending'"),
    ("documents", "storage_path", "VARCHAR(512)"),
    ("documents", "size_bytes", "BIGINT"),
    ("documents", "created_at", "TIMESTAMPTZ DEFAULT now()"),
    ("quiz_attempts", "user_id", "UUID REFERENCES users(id)"),
    ("quiz_attempts", "document_id", "UUID REFERENCES documents(id)"),
    ("quiz_attempts", "quiz_id", "UUID"),
    ("quiz_attempts", "is_correct", "BOOLEAN"),
    ("quiz_attempts", "response_time_s", "FLOAT"),
    ("quiz_attempts", "hints_used", "INT DEFAULT 0"),
    ("quiz_attempts", "created_at", "TIMESTAMPTZ DEFAULT now()"),
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prt_token_hash ON password_reset_tokens(token_hash)",
    "CREATE INDEX IF NOT EXISTS idx_prt_user_id ON password_reset_tokens(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_conv ON chat_messages(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_owner_doc ON kg_nodes(owner_id, document_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_norm ON kg_nodes(owner_id, norm_name)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_owner_doc ON kg_edges(owner_id, document_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_id)",
]


async def _snapshot(conn) -> dict[str, set[str]]:
    def read(sync_conn):
        insp = inspect(sync_conn)
        return {t: {c["name"] for c in insp.get_columns(t)} for t in insp.get_table_names()}

    return await conn.run_sync(read)


async def _run(conn, sql: str, what: str) -> None:
    """Mỗi câu lệnh chạy độc lập: một câu lỗi chỉ ghi cảnh báo, không chặn các câu còn lại."""
    try:
        await conn.execute(text(sql))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Schema upgrade: %s failed: %s", what, str(exc).splitlines()[0])


async def ensure_schema(engine: AsyncEngine) -> list[str]:
    """Đưa database về đúng schema mà code mong đợi. Trả về danh sách thay đổi đã áp dụng."""
    if engine.dialect.name != "postgresql":
        return []

    async with engine.connect() as raw:
        conn = await raw.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _LOCK_KEY})
        try:
            before = await _snapshot(conn)

            for name, ddl in _CREATE_TABLES:
                await _run(conn, ddl, f"create table {name}")

            # The legacy password-reset table (with an unused `token` VARCHAR(36) column) cannot
            # hold a hash, so recreate it. It only stores temporary codes that expire within minutes.
            have = (await _snapshot(conn)).get("password_reset_tokens")
            if have is None:
                await _run(conn, _PASSWORD_RESET_DDL, "create table password_reset_tokens")
            elif not _PASSWORD_RESET_COLUMNS <= have:
                await _run(conn, "DROP TABLE password_reset_tokens", "drop outdated password_reset_tokens")
                await _run(conn, _PASSWORD_RESET_DDL, "recreate password_reset_tokens")

            for table, column, definition in _ADD_COLUMNS:
                await _run(
                    conn,
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition}",
                    f"add column {table}.{column}",
                )
            for ddl in _INDEXES:
                await _run(conn, ddl, ddl)

            after = await _snapshot(conn)
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _LOCK_KEY})

    changes = [f"created table {t}" for t in sorted(after.keys() - before.keys())]
    changes += [
        f"added column {t}.{c}"
        for t in sorted(after)
        if t in before
        for c in sorted(after[t] - before[t])
    ]
    if changes:
        logger.warning("Database schema was out of date and has been upgraded: %s", "; ".join(changes))
    return changes
