# AI Study Assistant

Trợ lý học tập thông minh — RAG-based Q&A, tự động tạo quiz/flashcard, và
cá nhân hóa lộ trình học bằng Reinforcement Learning.

## Kiến trúc

- **frontend/** — ReactJS (Tailwind CSS)
- **backend-ai/** — FastAPI + AI Engine (RAG, RL) + Celery worker
- **postgres** — dữ liệu quan hệ (users, quiz, lịch sử làm bài)
- **qdrant** — vector store cho RAG
- **minio** — lưu trữ file tài liệu gốc
- **redis + celery** — xử lý bất đồng bộ (ingest tài liệu, chấm điểm nền)

## Chạy dự án

```bash
cp .env .env.local   # chỉnh sửa giá trị thật
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173
- Qdrant dashboard: http://localhost:6333/dashboard
- MinIO console: http://localhost:9001

## Ràng buộc chính

- Thời gian phản hồi mỗi truy vấn ≤ 800ms
- Không dùng web search tự do (chống hallucination — Grounded RAG only)
- Suy luận LLM phân tán trên 4 lõi GPU NVIDIA A16 độc lập, không tràn VRAM
- Toàn bộ kiến trúc đóng gói bằng Docker

## Đánh giá RAG bằng RAGAS

Tự động chấm faithfulness, answer_relevancy, context_precision, context_recall
(+ factual_correctness) và đo latency retrieval/generation trên bộ câu hỏi riêng.

```bash
# 1. Tạo backend-ai/eval/eval_set.jsonl từ eval_set.example.jsonl
#    (question, ground_truth, document_id = UUID tài liệu đã upload)
# 2. Chạy trong container backend (cần Ollama + Qdrant đang chạy)
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --tag hybrid
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --alpha 1.0 --tag dense-only
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --no-rerank --tag no-rerank
```

Báo cáo JSON + CSV nằm ở `backend-ai/eval_results/`. Judge mặc định là model Ollama
đang dùng; nên dùng model mạnh hơn bằng `RAGAS_JUDGE_MODEL` (hoặc
`RAGAS_JUDGE_PROVIDER=openai` + `RAGAS_JUDGE_BASE_URL` cho server tương thích OpenAI).
Model 3B làm judge thường cho nhiều dòng `failed` và điểm kém tin cậy.
