import { useState } from "react";
import { generateQuiz, submitQuizResult } from "../../services/api";

const STATUS = {
  IDLE: "idle",
  LOADING: "loading",
  PLAYING: "playing",
  FINISHED: "finished",
  ERROR: "error",
};

const DIFFICULTY_LABEL = {
  easy: { text: "Dễ", color: "#16a34a" },
  medium: { text: "Trung bình", color: "#d97706" },
  hard: { text: "Khó", color: "#dc2626" },
};

export default function QuizPlayer({ documentId = null }) {
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
        err.response?.data?.detail || err.message || "Không thể tạo quiz.";
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
    return (
      <div style={styles.container}>
        <h3 style={styles.heading}>Tạo bộ câu hỏi trắc nghiệm</h3>
        <input
          style={styles.input}
          placeholder="Chủ đề (ví dụ: Cấu trúc dữ liệu Stack)"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          disabled={status === STATUS.LOADING}
        />
        <label style={styles.label}>
          Số câu hỏi:
          <input
            style={styles.numberInput}
            type="number"
            min={1}
            max={50}
            value={numQuestions}
            onChange={(event) => setNumQuestions(Number(event.target.value))}
            disabled={status === STATUS.LOADING}
          />
        </label>
        <p style={styles.hint}>
          💡 Độ khó sẽ được điều chỉnh tự động theo kết quả làm bài của bạn.
        </p>
        <button
          style={styles.primaryButton}
          onClick={handleGenerate}
          disabled={status === STATUS.LOADING || !topic.trim()}
        >
          {status === STATUS.LOADING ? "Đang tạo…" : "Tạo Quiz"}
        </button>
        {errorMessage && <p style={styles.error}>{errorMessage}</p>}
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render: FINISHED
  // -------------------------------------------------------------------------
  if (status === STATUS.FINISHED) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div style={styles.container}>
        <h3 style={styles.heading}>Kết quả</h3>
        <p>
          Bạn trả lời đúng <strong>{score}/{questions.length}</strong> câu ({pct}%).
        </p>
        {nextDifficulty && (
          <p style={{ ...styles.difficultyBadge, color: DIFFICULTY_LABEL[nextDifficulty]?.color }}>
            Độ khó tiếp theo: <strong>{DIFFICULTY_LABEL[nextDifficulty]?.text ?? nextDifficulty}</strong>
          </p>
        )}
        <button style={styles.primaryButton} onClick={handleRestart}>
          Làm bộ câu hỏi khác
        </button>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render: PLAYING
  // -------------------------------------------------------------------------
  const diffLabel = currentDifficulty ? DIFFICULTY_LABEL[currentDifficulty] : null;

  return (
    <div style={styles.container}>
      {/* Header: progress + difficulty badge */}
      <div style={styles.headerRow}>
        <p style={styles.progress}>
          Câu {currentIndex + 1}/{questions.length} — Điểm: {score}
        </p>
        {diffLabel && (
          <span style={{ ...styles.badge, color: diffLabel.color, borderColor: diffLabel.color }}>
            {diffLabel.text}
          </span>
        )}
      </div>

      <h3 style={styles.question}>{currentQuestion.question}</h3>

      <div style={styles.optionList}>
        {currentQuestion.options.map((option, index) => {
          const isSelected = selectedOption === index;
          const isCorrectOption = index === currentQuestion.correct_index;
          let background = "#fff";
          if (isAnswerRevealed && isCorrectOption) background = "#dcfce7";
          else if (isAnswerRevealed && isSelected && !isCorrectOption) background = "#fee2e2";
          else if (isSelected) background = "#e0e7ff";

          return (
            <button
              key={index}
              style={{ ...styles.optionButton, backgroundColor: background }}
              onClick={() => handleSelectOption(index)}
              disabled={isAnswerRevealed}
            >
              {option}
            </button>
          );
        })}
      </div>

      {isAnswerRevealed && (
        <>
          <p style={styles.explanation}>{currentQuestion.explanation}</p>
          {/* Hiển thị phản hồi RL ngay sau khi submit câu */}
          {nextDifficulty && (
            <p style={{ ...styles.rlFeedback, color: DIFFICULTY_LABEL[nextDifficulty]?.color }}>
              {nextDifficulty === currentDifficulty
                ? `Giữ nguyên độ khó: ${DIFFICULTY_LABEL[nextDifficulty]?.text}`
                : nextDifficulty === "hard" || DIFFICULTY_LEVELS.indexOf(nextDifficulty) > DIFFICULTY_LEVELS.indexOf(currentDifficulty)
                  ? `🔼 Tăng lên: ${DIFFICULTY_LABEL[nextDifficulty]?.text}`
                  : `🔽 Giảm xuống: ${DIFFICULTY_LABEL[nextDifficulty]?.text}`}
            </p>
          )}
        </>
      )}

      <div style={styles.actionsRow}>
        {!isAnswerRevealed && (
          <button style={styles.secondaryButton} onClick={handleUseHint}>
            Dùng gợi ý ({hintsUsedThisQuestion})
          </button>
        )}
        {!isAnswerRevealed ? (
          <button
            style={styles.primaryButton}
            onClick={handleSubmitAnswer}
            disabled={selectedOption === null}
          >
            Trả lời
          </button>
        ) : (
          <button style={styles.primaryButton} onClick={handleNext}>
            {currentIndex + 1 >= questions.length ? "Xem kết quả" : "Câu tiếp theo"}
          </button>
        )}
      </div>
    </div>
  );
}

// Tham chiếu nội bộ cho tính toán hướng difficulty (tránh import vòng)
const DIFFICULTY_LEVELS = ["easy", "medium", "hard"];

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    padding: "16px",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
  },
  heading: { margin: 0 },
  headerRow: { display: "flex", alignItems: "center", justifyContent: "space-between" },
  label: { display: "flex", alignItems: "center", gap: "8px" },
  hint: { margin: 0, fontSize: "13px", color: "#64748b" },
  input: {
    padding: "8px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
  },
  numberInput: {
    width: "64px",
    padding: "4px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
  },
  primaryButton: {
    padding: "8px 16px",
    borderRadius: "6px",
    border: "none",
    backgroundColor: "#2563eb",
    color: "#fff",
    cursor: "pointer",
    alignSelf: "flex-start",
  },
  secondaryButton: {
    padding: "8px 16px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
    backgroundColor: "#fff",
    cursor: "pointer",
  },
  error: { color: "#dc2626" },
  progress: { color: "#64748b", margin: 0 },
  badge: {
    fontSize: "12px",
    fontWeight: 600,
    padding: "2px 8px",
    borderRadius: "999px",
    border: "1px solid",
  },
  question: { margin: 0 },
  optionList: { display: "flex", flexDirection: "column", gap: "8px" },
  optionButton: {
    textAlign: "left",
    padding: "10px 12px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
    cursor: "pointer",
  },
  explanation: {
    backgroundColor: "#f8fafc",
    padding: "10px",
    borderRadius: "6px",
    fontSize: "14px",
  },
  rlFeedback: {
    margin: 0,
    fontSize: "13px",
    fontWeight: 600,
  },
  difficultyBadge: {
    margin: 0,
    fontSize: "14px",
  },
  actionsRow: { display: "flex", gap: "8px" },
};
