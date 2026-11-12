import { useState } from "react";
import { generateQuiz, getErrorMessage, submitQuizResult } from "../../services/api";

const STATUS = {
  IDLE: "idle",
  LOADING: "loading",
  PLAYING: "playing",
  FINISHED: "finished",
  ERROR: "error",
};

const DIFFICULTY_LABEL = {
  easy: { text: "Dễ", tone: "ok" },
  medium: { text: "Trung bình", tone: "warn" },
  hard: { text: "Khó", tone: "err" },
};

export default function QuizPlayer({ documentId = null, scopeLabel = "Tất cả tài liệu" }) {
  const [topic, setTopic] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [status, setStatus] = useState(STATUS.IDLE);
  const [errorMessage, setErrorMessage] = useState(null);

  const [quizId, setQuizId] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);
  const [isAnswerRevealed, setIsAnswerRevealed] = useState(false);
  const [hintsUsedThisQuestion, setHintsUsedThisQuestion] = useState(0);
  const [questionStartedAt, setQuestionStartedAt] = useState(null);
  const [score, setScore] = useState(0);

  
  // Độ khó hiện tại (được backend xác định theo RL state của user)
  const [currentDifficulty, setCurrentDifficulty] = useState(null);
  // Độ khó tiếp theo (nhận về sau mỗi lần submit — dùng để hiển thị thông báo)
  const [nextDifficulty, setNextDifficulty] = useState(null);

  const handleGenerate = async () => {
    if (!topic.trim()) return;

    setStatus(STATUS.LOADING);
    setErrorMessage(null);
    setNextDifficulty(null);
    try {
      // Backend tự quyết định difficulty từ RL state — không cần client truyền lên
      const response = await generateQuiz(topic.trim(), numQuestions, documentId);
      const { quiz_id, questions: generatedQuestions, difficulty } = response.data;

      if (!generatedQuestions || generatedQuestions.length === 0) {
        throw new Error("Không có câu hỏi nào được tạo ra cho chủ đề này.");
      }

      setQuizId(quiz_id);
      setQuestions(generatedQuestions);
      setCurrentDifficulty(difficulty);   // ← lưu difficulty backend đã chọn
      setCurrentIndex(0);
      setScore(0);
      setSelectedOption(null);
      setIsAnswerRevealed(false);
      setHintsUsedThisQuestion(0);
      setQuestionStartedAt(Date.now());
      setStatus(STATUS.PLAYING);
    } catch (err) {
      const detail =
        getErrorMessage(err, err.message || "Không thể tạo quiz.");
      setErrorMessage(detail);
      setStatus(STATUS.ERROR);
    }
  };

  const currentQuestion = questions[currentIndex];

  const handleSelectOption = (optionIndex) => {
    if (isAnswerRevealed) return;
    setSelectedOption(optionIndex);
  };

  const handleUseHint = () => {
    if (isAnswerRevealed) return;
    setHintsUsedThisQuestion((prev) => prev + 1);
  };

  const handleSubmitAnswer = async () => {
    if (selectedOption === null || isAnswerRevealed) return;

    const isCorrect = selectedOption === currentQuestion.correct_index;
    const responseTimeS = questionStartedAt
      ? (Date.now() - questionStartedAt) / 1000
      : 0;

    setIsAnswerRevealed(true);
    if (isCorrect) setScore((prev) => prev + 1);

    try {
      const res = await submitQuizResult(quizId, {
        document_id: documentId,
        is_correct: isCorrect,
        response_time_s: responseTimeS,
        hints_used: hintsUsedThisQuestion,
      });
      // ← Lưu next_difficulty để hiển thị phản hồi cho user
      if (res.data?.next_difficulty) {
        setNextDifficulty(res.data.next_difficulty);
      }
    } catch {
      console.warn("Không thể gửi kết quả quiz lên server.");
    }
  };

  const handleNext = () => {
    if (currentIndex + 1 >= questions.length) {
      setStatus(STATUS.FINISHED);
      return;
    }
    setCurrentIndex((prev) => prev + 1);
    setSelectedOption(null);
    setIsAnswerRevealed(false);
    setHintsUsedThisQuestion(0);
    setNextDifficulty(null);
    setQuestionStartedAt(Date.now());
  };

  const handleRestart = () => {
    setStatus(STATUS.IDLE);
    setQuestions([]);
    setQuizId(null);
    setCurrentDifficulty(null);
    setNextDifficulty(null);
  };

  // -------------------------------------------------------------------------
  // Render: IDLE / LOADING / ERROR
  // -------------------------------------------------------------------------
  if (status === STATUS.IDLE || status === STATUS.LOADING || status === STATUS.ERROR) {
    const isLoading = status === STATUS.LOADING;
    return (
      <div className="page">
        <header className="page-head">
          <h1 className="page-title">Tạo bộ câu hỏi trắc nghiệm</h1>
          <p className="page-sub">Nhập chủ đề, câu hỏi sẽ được tạo từ nội dung tài liệu của bạn.</p>
        </header>

        <form
          className="card card-pad"
          style={{ display: "grid", gap: 18 }}
          onSubmit={(event) => {
            event.preventDefault();
            if (!isLoading) handleGenerate();
          }}
        >
          <div className="field">
            <label className="label" htmlFor="quiz-topic">
              Chủ đề
            </label>
            <input
              id="quiz-topic"
              className="input"
              placeholder="Ví dụ: Cấu trúc dữ liệu Stack"
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              disabled={isLoading}
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="quiz-count">
              Số câu hỏi
            </label>
            <input
              id="quiz-count"
              className="input input--num"
              type="number"
              min={1}
              max={50}
              value={numQuestions}
              onChange={(event) => setNumQuestions(Number(event.target.value))}
              disabled={isLoading}
            />
          </div>
          <p className="field-hint">
            Phạm vi: {scopeLabel}. Độ khó tự điều chỉnh theo kết quả làm bài của bạn.
          </p>
          {errorMessage && (
            <p className="note note--err" role="alert">
              {errorMessage}
            </p>
          )}
          <div className="actions">
            <button type="submit" className="btn btn--primary btn--lg" disabled={isLoading || !topic.trim()}>
              {isLoading ? "Đang tạo…" : "Tạo quiz"}
            </button>
          </div>
        </form>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render: FINISHED
  // -------------------------------------------------------------------------
  if (status === STATUS.FINISHED) {
    const pct = Math.round((score / questions.length) * 100);
    const next = nextDifficulty ? DIFFICULTY_LABEL[nextDifficulty] : null;
    return (
      <div className="page">
        <header className="page-head">
          <h1 className="page-title">Kết quả</h1>
          <p className="page-sub">Bạn đã hoàn thành bộ câu hỏi.</p>
        </header>
        <section className="card card-pad" style={{ display: "grid", gap: 16 }}>
          <div>
            <div className="result-score">
              {score}/{questions.length}
            </div>
            <p className="page-sub">Trả lời đúng {pct}% số câu.</p>
          </div>
          {next && (
            <p>
              Độ khó tiếp theo: <span className={`badge badge--${next.tone}`}>{next.text}</span>
            </p>
          )}
          <div className="actions">
            <button type="button" className="btn btn--primary btn--lg" onClick={handleRestart}>
              Làm bộ câu hỏi khác
            </button>
          </div>
        </section>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render: PLAYING
  // -------------------------------------------------------------------------
  const diffLabel = currentDifficulty ? DIFFICULTY_LABEL[currentDifficulty] : null;
  const answered = currentIndex + (isAnswerRevealed ? 1 : 0);

  return (
    <div className="page">
      <div className="quiz-meta">
        <span>
          Câu {currentIndex + 1}/{questions.length}
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
          {diffLabel && <span className={`badge badge--${diffLabel.tone}`}>{diffLabel.text}</span>}
          Điểm: {score}
        </span>
      </div>
      <div
        className="progress"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={questions.length}
        aria-valuenow={answered}
      >
        <div style={{ width: `${(answered / questions.length) * 100}%` }} />
      </div>

      <h2 className="question">{currentQuestion.question}</h2>

      <div className="options">
        {currentQuestion.options.map((option, index) => {
          const isSelected = selectedOption === index;
          const isCorrectOption = index === currentQuestion.correct_index;
          let tone = "";
          if (isAnswerRevealed && isCorrectOption) tone = " option--correct";
          else if (isAnswerRevealed && isSelected && !isCorrectOption) tone = " option--wrong";
          else if (isSelected) tone = " option--selected";

          return (
            <button
              key={index}
              type="button"
              className={`option${tone}`}
              aria-pressed={isSelected}
              onClick={() => handleSelectOption(index)}
              disabled={isAnswerRevealed}
            >
              <span className="option-key">{String.fromCharCode(65 + index)}</span>
              <span>{option}</span>
            </button>
          );
        })}
      </div>

      {isAnswerRevealed && (
        <>
          <p className="explain">{currentQuestion.explanation}</p>
          {/* Hiển thị phản hồi RL ngay sau khi submit câu */}
          {nextDifficulty && (
            <p className="field-hint" style={{ marginTop: 10 }}>
              {nextDifficulty === currentDifficulty
                ? `Giữ nguyên độ khó: ${DIFFICULTY_LABEL[nextDifficulty]?.text}`
                : nextDifficulty === "hard" || DIFFICULTY_LEVELS.indexOf(nextDifficulty) > DIFFICULTY_LEVELS.indexOf(currentDifficulty)
                  ? `Tăng lên: ${DIFFICULTY_LABEL[nextDifficulty]?.text}`
                  : `Giảm xuống: ${DIFFICULTY_LABEL[nextDifficulty]?.text}`}
            </p>
          )}
        </>
      )}

      <div className="quiz-actions">
        {!isAnswerRevealed && (
          <button type="button" className="btn btn--lg" onClick={handleUseHint}>
            Dùng gợi ý ({hintsUsedThisQuestion})
          </button>
        )}
        {!isAnswerRevealed ? (
          <button
            type="button"
            className="btn btn--primary btn--lg"
            onClick={handleSubmitAnswer}
            disabled={selectedOption === null}
          >
            Trả lời
          </button>
        ) : (
          <button type="button" className="btn btn--primary btn--lg" onClick={handleNext}>
            {currentIndex + 1 >= questions.length ? "Xem kết quả" : "Câu tiếp theo"}
          </button>
        )}
      </div>
    </div>
  );
}

// Tham chiếu nội bộ cho tính toán hướng difficulty (tránh import vòng)
const DIFFICULTY_LEVELS = ["easy", "medium", "hard"];
