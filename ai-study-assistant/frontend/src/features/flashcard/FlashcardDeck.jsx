import { useState } from "react";
import { generateQuiz, getErrorMessage } from "../../services/api";

const RECALL_LEVELS = [
  { key: "again", label: "Chưa nhớ" },
  { key: "hard", label: "Khó" },
  { key: "good", label: "Tốt" },
  { key: "easy", label: "Dễ" },
];

export default function FlashcardDeck({ documentId = null, scopeLabel = "Tất cả tài liệu" }) {
  const [topic, setTopic] = useState("");
  const [cards, setCards] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [recallCounts, setRecallCounts] = useState({});

  const handleGenerate = async () => {
    if (!topic.trim() || isLoading) return;

    setIsLoading(true);
    setErrorMessage(null);
    try {
      const response = await generateQuiz(topic.trim(), 10, documentId);
      const generatedCards = (response.data.questions || []).map((q) => ({
        front: q.question,
        back: `${q.options[q.correct_index]}\n\n${q.explanation}`,
      }));

      if (generatedCards.length === 0) {
        throw new Error("Không tạo được thẻ nào cho chủ đề này.");
      }

      setCards(generatedCards);
      setCurrentIndex(0);
      setIsFlipped(false);
      setRecallCounts({});
    } catch (err) {
      const detail =
        getErrorMessage(err, err.message || "Không thể tạo bộ thẻ ghi nhớ.");
      setErrorMessage(detail);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFlip = () => setIsFlipped((prev) => !prev);

  const handleRate = (levelKey) => {
    setRecallCounts((prev) => ({ ...prev, [levelKey]: (prev[levelKey] || 0) + 1 }));

    // Luôn sang thẻ kế tiếp; chấm thẻ cuối cùng chuyển sang màn hình tổng kết.
    setCurrentIndex((prev) => prev + 1);
    setIsFlipped(false);
  };

  if (cards.length === 0) {
    return (
      <div className="page">
        <header className="page-head">
          <h1 className="page-title">Tạo bộ thẻ ghi nhớ</h1>
          <p className="page-sub">Nhập chủ đề, hệ thống sẽ tạo 10 thẻ hỏi đáp từ tài liệu của bạn.</p>
        </header>

        <form
          className="card card-pad"
          style={{ display: "grid", gap: 18 }}
          onSubmit={(event) => {
            event.preventDefault();
            handleGenerate();
          }}
        >
          <div className="field">
            <label className="label" htmlFor="fc-topic">
              Chủ đề
            </label>
            <input
              id="fc-topic"
              className="input"
              placeholder="Ví dụ: Từ vựng mạng máy tính"
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              disabled={isLoading}
            />
          </div>
          <p className="field-hint">Phạm vi: {scopeLabel}</p>
          {errorMessage && (
            <p className="note note--err" role="alert">
              {errorMessage}
            </p>
          )}
          <div className="actions">
            <button type="submit" className="btn btn--primary btn--lg" disabled={isLoading || !topic.trim()}>
              {isLoading ? "Đang tạo…" : "Tạo thẻ ghi nhớ"}
            </button>
          </div>
        </form>
      </div>
    );
  }

  const isDeckFinished = currentIndex >= cards.length;
  const currentCard = cards[Math.min(currentIndex, cards.length - 1)];

  return (
    <div className="page">
      <div className="quiz-meta">
        <span>
          Thẻ {Math.min(currentIndex + 1, cards.length)}/{cards.length}
        </span>
        <button
          type="button"
          className="btn"
          onClick={() => {
            setCards([]);
            setCurrentIndex(0);
            setIsFlipped(false);
            setRecallCounts({});
          }}
        >
          Bộ thẻ mới
        </button>
      </div>
      <div className="progress">
        <div style={{ width: `${(Math.min(currentIndex, cards.length) / cards.length) * 100}%` }} />
      </div>

      {isDeckFinished ? (
        <section className="card card-pad">
          <h2 className="card-title">Bạn đã ôn hết bộ thẻ này</h2>
          <ul className="summary">
            {RECALL_LEVELS.map((level) => (
              <li key={level.key}>
                <span>{level.label}</span>
                <b>{recallCounts[level.key] || 0}</b>
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="btn btn--primary btn--lg"
            onClick={() => {
              setCurrentIndex(0);
              setIsFlipped(false);
              setRecallCounts({});
            }}
          >
            Ôn lại từ đầu
          </button>
        </section>
      ) : (
        <>
          <div className="flip" data-flipped={isFlipped}>
            <div className="flip-inner">
              <div
                className="flip-face"
                role="button"
                tabIndex={isFlipped ? -1 : 0}
                aria-hidden={isFlipped}
                aria-label="Mặt trước: lật thẻ để xem đáp án"
                onClick={handleFlip}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), handleFlip())}
              >
                <p>{currentCard.front}</p>
              </div>
              <div
                className="flip-face flip-face--back"
                role="button"
                tabIndex={isFlipped ? 0 : -1}
                aria-hidden={!isFlipped}
                aria-label="Mặt sau: lật thẻ để xem lại câu hỏi"
                onClick={handleFlip}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), handleFlip())}
              >
                <p>{currentCard.back}</p>
              </div>
            </div>
          </div>
          <p className="flip-hint">{isFlipped ? "Bấm vào thẻ để xem lại câu hỏi" : "Bấm vào thẻ để xem đáp án"}</p>

          {isFlipped && (
            <div className="rating">
              {RECALL_LEVELS.map((level) => (
                <button key={level.key} type="button" className="btn btn--lg" onClick={() => handleRate(level.key)}>
                  {level.label}
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
