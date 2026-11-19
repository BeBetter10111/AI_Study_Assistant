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
cp .env .env.local   # Edit with your actual values.
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
# 1. Create backend-ai/eval/eval_set.jsonl from eval_set.example.jsonl
#    (question, ground_truth, document_id = UUID of an uploaded document)
# 2. Run inside the backend container (Ollama + Qdrant must be running)
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --tag hybrid
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --alpha 1.0 --tag dense-only
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --no-rerank --tag no-rerank
```

Báo cáo JSON + CSV nằm ở `backend-ai/eval_results/`. Judge mặc định là model Ollama
đang dùng; nên dùng model mạnh hơn bằng `RAGAS_JUDGE_MODEL` (hoặc
`RAGAS_JUDGE_PROVIDER=openai` + `RAGAS_JUDGE_BASE_URL` cho server tương thích OpenAI).
Model 3B làm judge thường cho nhiều dòng `failed` và điểm kém tin cậy.


## Tính năng học tập mở rộng

| Tính năng | Giao diện | API | Ghi chú |
|---|---|---|---|
| Kiểm chứng sau sinh + độ tin cậy | Huy hiệu dưới câu trả lời trong Chat | trường `confidence`, `confidence_label`, `unsupported_claims` của `/api/chat/query[/stream]` | Heuristic groundedness (không gọi thêm LLM); tắt bằng `VERIFICATION_ENABLED=false` |
| Chế độ ôn thi | Ôn thi | `/api/exams/*` | Đề tạo từ đề cương, **chấm điểm trên máy chủ**, báo cáo chủ đề yếu (làm mượt Beta + khoảng Wilson) |
| Chế độ giảng viên | Giảng viên | `/api/lecturer/*` | Ngân hàng câu hỏi + rubric, ghép đề (phân tầng độ khó), xuất Word/Markdown, chấm thử theo rubric (AI chỉ gợi ý) |
| Luyện nói | Luyện nói | `/api/speech/*` | Whisper (faster-whisper) nhận dạng, WER so với câu mẫu, tốc độ/ngắt nghỉ/độ rõ; nghe mẫu bằng TTS của trình duyệt |

Cấu hình giọng nói (tuỳ chọn): `pip install faster-whisper`, `STT_ENABLED=true`, `WHISPER_MODEL=small|medium|large-v3`,
`WHISPER_DEVICE=auto|cpu|cuda`. Lần đầu dùng sẽ tải model; thiếu thư viện thì `/api/speech/*` trả 503 kèm hướng dẫn.
Ghi âm cần HTTPS hoặc `localhost`. Bản ghi chỉ xử lý trong bộ nhớ, không lưu.

Bảng mới (`exams`, `exam_questions`, `exam_attempts`, `question_bank_items`, `exam_papers`) được tạo tự động khi khởi động;
nếu tắt `AUTO_MIGRATE_SCHEMA` hãy chạy `scripts/migrations/005_exam_lecturer.sql`.
