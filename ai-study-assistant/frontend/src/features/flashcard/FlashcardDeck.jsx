import { useState } from "react";
import { generateQuiz } from "../../services/api";

const RECALL_LEVELS = [
  { key: "again", label: "Not remembered" },
  { key: "hard", label: "Difficult" },
  { key: "good", label: "Good" },
  { key: "easy", label: "Easy" },
];

export default function FlashcardDeck({ documentId = null }) {
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
        throw new Error("No flashcards generated for this topic.");
      }

      setCards(generatedCards);
      setCurrentIndex(0);
      setIsFlipped(false);
      setRecallCounts({});
    } catch (err) {
      const detail =
        err.response?.data?.detail || err.message || "Unable to generate flashcard deck.";
      setErrorMessage(detail);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFlip = () => setIsFlipped((prev) => !prev);

  const handleRate = (levelKey) => {
    setRecallCounts((prev) => ({ ...prev, [levelKey]: (prev[levelKey] || 0) + 1 }));

    if (currentIndex + 1 < cards.length) {
      setCurrentIndex((prev) => prev + 1);
      setIsFlipped(false);
    }
  };

  if (cards.length === 0) {
    return (
      <div style={styles.container}>
        <h3 style={styles.heading}>Tạo bộ thẻ ghi nhớ</h3>
        <input
          style={styles.input}
          placeholder="Topic (e.g., Computer Networking Vocabulary)"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          disabled={isLoading}
        />
        <button
          style={styles.primaryButton}
          onClick={handleGenerate}
          disabled={isLoading || !topic.trim()}
        >
          {isLoading ? "Generating…" : "Generate Flashcards"}
        </button>
        {errorMessage && <p style={styles.error}>{errorMessage}</p>}
      </div>
    );
  }

  const isDeckFinished = currentIndex >= cards.length;
  const currentCard = cards[Math.min(currentIndex, cards.length - 1)];

  return (
    <div style={styles.container}>
      <p style={styles.progress}>
        Thẻ {Math.min(currentIndex + 1, cards.length)}/{cards.length}
      </p>

      {isDeckFinished ? (
        <div>
          <p>Bạn đã ôn hết bộ thẻ này.</p>
          <ul style={styles.recallSummary}>
            {RECALL_LEVELS.map((level) => (
              <li key={level.key}>
                {level.label}: {recallCounts[level.key] || 0}
              </li>
            ))}
          </ul>
          <button
            style={styles.primaryButton}
            onClick={() => {
              setCurrentIndex(0);
              setIsFlipped(false);
            }}
          >
            Ôn lại từ đầu
          </button>
        </div>
      ) : (
        <>
          <div style={styles.card} onClick={handleFlip}>
            <p style={styles.cardText}>{isFlipped ? currentCard.back : currentCard.front}</p>
            <span style={styles.flipHint}>
              {isFlipped ? "Click to view question" : "Click to flip card"}
            </span>
          </div>

          {isFlipped && (
            <div style={styles.ratingRow}>
              {RECALL_LEVELS.map((level) => (
                <button
                  key={level.key}
                  style={styles.ratingButton}
                  onClick={() => handleRate(level.key)}
                >
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
  input: {
    padding: "8px",
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
  error: { color: "#dc2626" },
  progress: { color: "#64748b", margin: 0 },
  recallSummary: { margin: "8px 0", paddingLeft: "20px" },
  card: {
    minHeight: "160px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    borderRadius: "10px",
    backgroundColor: "#f8fafc",
    border: "1px solid #cbd5e1",
    cursor: "pointer",
    textAlign: "center",
    gap: "12px",
  },
  cardText: { margin: 0, whiteSpace: "pre-wrap" },
  flipHint: { fontSize: "12px", color: "#94a3b8" },
  ratingRow: { display: "flex", gap: "8px", flexWrap: "wrap" },
  ratingButton: {
    flex: 1,
    padding: "8px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
    backgroundColor: "#fff",
    cursor: "pointer",
  },
};
