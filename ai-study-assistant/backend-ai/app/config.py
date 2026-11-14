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

    # --- Mô hình embedding / re-ranker: ONNX Runtime INT8 
    # Tên model gốc (HuggingFace). Đổi embedding model => phải re-ingest tài liệu.
    rag_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # Dùng bản ONNX đã lượng tử hóa nếu có file (tạo bằng `python -m scripts.quantize_models`).
    # Không có file / lỗi tải => TỰ ĐỘNG quay về PyTorch (hành vi cũ), không làm hỏng API.
    rag_onnx_enabled: bool = True
    rag_onnx_dir: str = "models/onnx"         # tương đối so với /app; nằm trong volume ./backend-ai
    rag_onnx_precision: str = "int8"          # "int8" (nhanh, nhẹ) hoặc "fp32" (ONNX không lượng tử hóa)
    rag_onnx_threads: int = 0                 # 0 = để ONNX Runtime tự chọn theo số lõi CPU
    # Cache LRU cho embedding của câu truy vấn (cùng câu hỏi => khỏi tính lại).
    rag_query_embed_cache_size: int = 256

    # --- Redis semantic cache
    semantic_cache_enabled: bool = True
    # Ngưỡng cosine giữa 2 câu hỏi để coi là "cùng một câu hỏi". Cao = an toàn hơn (ít trúng sai).
    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_s: int = 24 * 3600
    semantic_cache_max_entries: int = 200     # tối đa mỗi phạm vi (user + tài liệu + intent + top_k)
    semantic_cache_timeout_s: float = 0.5     # Redis chậm/chết => bỏ qua cache, KHÔNG làm chậm chat
    # Heartbeat SSE (giây) khi chờ token đầu tiên, tránh proxy cắt kết nối "im lặng".
    sse_heartbeat_s: float = 15.0

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
