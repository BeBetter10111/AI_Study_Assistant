import { useState } from "react";
import { generateQuiz, submitQuizResult } from "../../services/api";

const STATUS = {
  IDLE: "idle",
  LOADING: "loading",
  PLAYING: "playing",
  FINISHED: "finished",
  ERROR: "error",
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

  const handleGenerate = async () => {
    if (!topic.trim()) return;

    setStatus(STATUS.LOADING);
    setErrorMessage(null);
    try {
      const response = await generateQuiz(topic.trim(), numQuestions, documentId);
      const { quiz_id, questions: generatedQuestions } = response.data;

      if (!generatedQuestions || generatedQuestions.length === 0) {
        throw new Error("No questions generated for this topic.");
      }

      setQuizId(quiz_id);
      setQuestions(generatedQuestions);
      setCurrentIndex(0);
      setScore(0);
      setSelectedOption(null);
      setIsAnswerRevealed(false);
      setHintsUsedThisQuestion(0);
      setQuestionStartedAt(Date.now());
      setStatus(STATUS.PLAYING);
    } catch (err) {
      const detail =
        err.response?.data?.detail || err.message || "Unable to generate quiz.";
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
      await submitQuizResult(quizId, {
        document_id: documentId,
        is_correct: isCorrect,
        response_time_s: responseTimeS,
        hints_used: hintsUsedThisQuestion,
      });
    } catch {
      console.warn("Unable to submit quiz result to the server.");
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
    setQuestionStartedAt(Date.now());
  };

  const handleRestart = () => {
    setStatus(STATUS.IDLE);
    setQuestions([]);
    setQuizId(null);
  };

  if (status === STATUS.IDLE || status === STATUS.LOADING || status === STATUS.ERROR) {
    return (
      <div style={styles.container}>
        <h3 style={styles.heading}>Tạo bộ câu hỏi trắc nghiệm</h3>
        <input
          style={styles.input}
          placeholder="Topic (e.g., Stack Data Structure)"
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
        <button
          style={styles.primaryButton}
          onClick={handleGenerate}
          disabled={status === STATUS.LOADING || !topic.trim()}
        >
          {status === STATUS.LOADING ? "Generating…" : "Generate Quiz"}
        </button>
        {errorMessage && <p style={styles.error}>{errorMessage}</p>}
      </div>
    );
  }

  if (status === STATUS.FINISHED) {
    return (
      <div style={styles.container}>
        <h3 style={styles.heading}>Results</h3>
        <p>
          Bạn trả lời đúng {score}/{questions.length} câu.
        </p>
        <button style={styles.primaryButton} onClick={handleRestart}>
          Làm bộ câu hỏi khác
        </button>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <p style={styles.progress}>
        Câu {currentIndex + 1}/{questions.length} — Điểm: {score}
      </p>
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
        <p style={styles.explanation}>{currentQuestion.explanation}</p>
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
            {currentIndex + 1 >= questions.length ? "View Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

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
  label: { display: "flex", alignItems: "center", gap: "8px" },
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
  actionsRow: { display: "flex", gap: "8px" },
};
