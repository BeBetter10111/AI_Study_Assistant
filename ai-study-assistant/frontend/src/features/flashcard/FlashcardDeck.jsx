import { useState } from "react";
import { generateQuiz, getErrorMessage } from "../../services/api";

const RECALL_LEVELS = [
  { key: "again", label: "Forgot" },
  { key: "hard", label: "Hard" },
  { key: "good", label: "Good" },
  { key: "easy", label: "Easy" },
];

export default function FlashcardDeck({ documentId = null, scopeLabel = "All documents" }) {
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
        throw new Error("No cards could be created for this topic.");
      }

      setCards(generatedCards);
      setCurrentIndex(0);
      setIsFlipped(false);
      setRecallCounts({});
    } catch (err) {
      const detail =
        getErrorMessage(err, err.message || "Could not create the flashcards.");
      setErrorMessage(detail);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFlip = () => setIsFlipped((prev) => !prev);

  const handleRate = (levelKey) => {
    setRecallCounts((prev) => ({ ...prev, [levelKey]: (prev[levelKey] || 0) + 1 }));

    // Always advance to the next card; rating the last card opens the summary screen.
    setCurrentIndex((prev) => prev + 1);
    setIsFlipped(false);
  };

  if (cards.length === 0) {
    return (
      <div className="page">
        <header className="page-head">
          <h1 className="page-title">Create flashcards</h1>
          <p className="page-sub">Enter a topic and the system will create 10 question-and-answer cards from your documents.</p>
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
              Topic
            </label>
            <input
              id="fc-topic"
              className="input"
              placeholder="Example: Computer network terminology"
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              disabled={isLoading}
            />
          </div>
          <p className="field-hint">Scope: {scopeLabel}</p>
          {errorMessage && (
            <p className="note note--err" role="alert">
              {errorMessage}
            </p>
          )}
          <div className="actions">
            <button type="submit" className="btn btn--primary btn--lg" disabled={isLoading || !topic.trim()}>
              {isLoading ? "Generating…" : "Create flashcards"}
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
          Card {Math.min(currentIndex + 1, cards.length)}/{cards.length}
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
          New deck
        </button>
      </div>
      <div className="progress">
        <div style={{ width: `${(Math.min(currentIndex, cards.length) / cards.length) * 100}%` }} />
      </div>

      {isDeckFinished ? (
        <section className="card card-pad">
          <h2 className="card-title">You finished this deck</h2>
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
            Review from the beginning
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
                aria-label="Front: flip card to see answer"
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
                aria-label="Back: flip card to see question"
                onClick={handleFlip}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), handleFlip())}
              >
                <p>{currentCard.back}</p>
              </div>
            </div>
          </div>

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
