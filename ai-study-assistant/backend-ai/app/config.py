from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://asa:asa_password@postgres:5432/asa_db"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "asa_documents"

    minio_endpoint: str = "minio:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "asa-documents"

    redis_url: str = "redis://redis:6379/0"
    auto_migrate_schema: bool = True

    llm_backend: str = "vllm"
    llm_model_name: str = "change-me"
    llm_tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.85

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model_name: str = "qwen2.5:7b"
    ollama_num_ctx: int = 8192
    ollama_timeout_s: float = 300.0

    max_response_latency_ms: int = 800
    jwt_secret: str = "change-me"
    app_env: str = "production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Password reset (forgot-password flow)
    password_reset_expire_minutes: int = 15       # lifetime of the emailed code
    password_reset_session_minutes: int = 10      # lifetime of the reset session after the code is verified
    password_reset_resend_cooldown_s: int = 60    # min gap between two code requests for one account
    password_reset_max_attempts: int = 5          # wrong-code attempts before the code is burned

    # SMTP (leave SMTP_HOST empty in dev: the code is written to the backend log instead)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""            
    smtp_use_starttls: bool = True  
    smtp_use_ssl: bool = False      

    # RAG retrieval tuning
    # Candidates fetched PER LEG (dense and BM25) before fusion.
    rag_fetch_k: int = 20
    # Weight of the dense leg in the RRF fusion: 1.0 = pure dense, 0.0 = pure BM25.
    rag_hybrid_alpha: float = 0.7
    # RRF constant (higher = flatter contribution of lower ranks).
    rag_rrf_k: int = 60
    # Index/search word bigrams in the BM25 leg (helps multi-syllable Vietnamese words).
    # Changing this requires re-ingesting documents.
    rag_sparse_bigrams: bool = True
    # Set to false to skip the cross-encoder re-ranker (faster but lower quality).
    rag_rerank_enabled: bool = True
    # Multilingual cross-encoder (covers Vietnamese); ms-marco models are English-only.
    rag_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # How many fused candidates are sent to the cross-encoder.
    rag_rerank_top_n: int = 20
    # Final score = w * rerank + (1 - w) * fusion (both min-max normalised).
    rag_rerank_weight: float = 0.7

    # --- Model embedding / re-ranker: ONNX Runtime INT8 
    rag_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    rag_onnx_enabled: bool = True
    rag_onnx_dir: str = "models/onnx"         # Relative to /app; stored in the ./backend-ai volume.
    rag_onnx_precision: str = "int8"          # "int8" (faster, smaller) or "fp32" (unquantized ONNX).
    rag_onnx_threads: int = 0                 # 0 = let ONNX Runtime choose based on the number of CPU cores.
    # LRU cache for query embeddings (repeated questions do not need to be recomputed).
    rag_query_embed_cache_size: int = 256

    # --- Redis semantic cache
    semantic_cache_enabled: bool = True
    # Cosine-similarity threshold for treating two questions as the same. Higher = safer (fewer false matches).
    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_s: int = 24 * 3600
    semantic_cache_max_entries: int = 200     # Maximum per scope (user + document + intent + top_k).
    semantic_cache_timeout_s: float = 0.5     # If Redis is slow/unavailable, skip the cache; do NOT slow down chat.
    # SSE heartbeat interval (seconds) while waiting for the first token, to prevent proxies from closing an idle connection.
    sse_heartbeat_s: float = 15.0

    # --- Load limits / abuse prevention
    # Maximum concurrent Q&A requests per user (exceeding this returns 429). Each stream holds one thread.
    max_concurrent_requests_per_user: int = 2
    # Maximum concurrent LLM calls across the backend (chat, quiz, etc.). When exceeded, report "busy" instead of queuing
    # indefinitely, exhausting the thread pool (default 40) and blocking logins. 0 = unlimited.
    max_concurrent_generations: int = 4
    # Upload: concurrent MinIO writes, maximum wait time for a slot, and per-user limits.
    max_concurrent_uploads: int = 3
    upload_slot_wait_s: float = 30.0
    max_documents_per_user: int = 1000
    max_storage_per_user_mb: int = 3 * 1024
    # Login: after this many FAILED attempts in the window, soft-lock the account — allow at most one attempt
    # every `login_soft_lock_interval_s` seconds (from any IP). A successful login clears the counter. 0 = disabled.
    login_fail_threshold: int = 8
    login_fail_window_s: int = 900
    login_soft_lock_interval_s: int = 15
    #Each user can have at most this many conversations, and each conversation can have at most this many messages. Conversations that are empty for more than `empty_conversation_ttl_s` seconds will be purged automatically.
    max_conversations_per_user: int = 500
    max_messages_per_conversation: int = 500
    empty_conversation_ttl_s: int = 24 * 3600

    allowed_origins: str = "http://localhost:5173"

    # RAGAS evaluation 
    # Judge LLM: "ollama" (default, reuses OLLAMA_BASE_URL) or "openai" (any
    # OpenAI-compatible server: vLLM, OpenAI, ...).  A judge stronger than the
    # model being evaluated gives far more trustworthy scores.
    ragas_judge_provider: str = "ollama"
    ragas_judge_model: str = ""          # empty -> OLLAMA_MODEL_NAME
    ragas_judge_base_url: str = ""       # empty -> OLLAMA_BASE_URL (ollama) / OpenAI default (openai)
    ragas_judge_api_key: str = "not-needed"
    ragas_max_workers: int = 2           # parallel judge calls; keep low for a local GPU
    ragas_timeout_s: float = 600.0       # per judge call
    ragas_output_dir: str = "eval_results"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
