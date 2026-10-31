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

    llm_backend: str = "vllm"
    llm_model_name: str = "change-me"
    llm_tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.85

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model_name: str = "qwen2.5:3b"
    ollama_num_ctx: int = 8192
    ollama_timeout_s: float = 300.0

    max_response_latency_ms: int = 800
    jwt_secret: str = "change-me"
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
    smtp_from: str = ""            # e.g. "AI Study Assistant <no-reply@example.com>"
    smtp_use_starttls: bool = True  # port 587
    smtp_use_ssl: bool = False      # port 465 (implicit TLS)

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

    # RAGAS evaluation (python -m app.rag.evaluation) — never used by the API itself.
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
