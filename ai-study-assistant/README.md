# AI Study Assistant

An intelligent study assistant featuring RAG-based Q&A, automatic quiz and flashcard
generation, and personalized learning paths using Reinforcement Learning.

## Architecture

- **frontend/** — React.js (Tailwind CSS)
- **backend-ai/** — FastAPI + AI Engine (RAG, RL) + Celery worker
- **postgres** — relational data (users, quizzes, and quiz history)
- **qdrant** — vector store for RAG
- **minio** — storage for original document files
- **redis + celery** — asynchronous processing (document ingestion and background scoring)

## Run the Project

```bash
cp .env .env.local   # Edit with your actual values.
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173
- Qdrant dashboard: http://localhost:6333/dashboard
- MinIO console: http://localhost:9001

## Main Constraints

- Query response time is ≤ 800 ms
- No unrestricted web search (Grounded RAG only, to reduce hallucinations)
- LLM inference is distributed across four independent NVIDIA A16 GPU cores without VRAM overflow
- The entire architecture is packaged with Docker

## RAG Evaluation with RAGAS

Automatically evaluates faithfulness, answer_relevancy, context_precision, context_recall,
and factual_correctness, and measures retrieval and generation latency on a dedicated question set.

```bash
# 1. Create backend-ai/eval/eval_set.jsonl from eval_set.example.jsonl
#    (question, ground_truth, document_id = UUID of an uploaded document)
# 2. Run inside the backend container (Ollama + Qdrant must be running)
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --tag hybrid
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --alpha 1.0 --tag dense-only
docker compose exec backend-ai python -m app.rag.evaluation eval/eval_set.jsonl --no-rerank --tag no-rerank
```

JSON and CSV reports are stored in `backend-ai/eval_results/`. The default judge is the
currently used Ollama model. A stronger model can be selected with `RAGAS_JUDGE_MODEL`,
or with `RAGAS_JUDGE_PROVIDER=openai` and `RAGAS_JUDGE_BASE_URL` for an
OpenAI-compatible server. A 3B model used as the judge commonly produces failed
entries and less reliable scores.


## Extended Learning Features

| Feature | Interface | API | Notes |
|---|---|---|---|
| Post-answer verification and confidence | Badge below the chat answer | `confidence`, `confidence_label`, and `unsupported_claims` fields in `/api/chat/query[/stream]` | Heuristic groundedness check (no additional LLM call); disable with `VERIFICATION_ENABLED=false` |
| Exam review mode | Exam Review | `/api/exams/*` | Exams generated from outlines, **server-side grading**, weak-topic reports (Beta smoothing and Wilson interval) |
| Lecturer mode | Lecturer | `/api/lecturer/*` | Question bank and rubrics, exam paper assembly with difficulty levels, Word/Markdown export, and rubric-based review suggestions (AI suggestions only) |
| Speaking practice | Speaking Practice | `/api/speech/*` | Whisper speech recognition (faster-whisper), WER comparison with reference sentences, and speed/pause/clarity analysis; browser TTS for listening samples |

Optional speech configuration: `pip install faster-whisper`, `STT_ENABLED=true`,
`WHISPER_MODEL=small|medium|large-v3`, and `WHISPER_DEVICE=auto|cpu|cuda`.
The model is downloaded the first time it is used. If the library is missing,
`/api/speech/*` returns 503 with setup instructions. Recording requires HTTPS or
`localhost`. Recordings are processed in memory and are not saved.

The new tables (`exams`, `exam_questions`, `exam_attempts`, `question_bank_items`,
and `exam_papers`) are created automatically at startup. If `AUTO_MIGRATE_SCHEMA`
is disabled, run `scripts/migrations/005_exam_lecturer.sql`.
