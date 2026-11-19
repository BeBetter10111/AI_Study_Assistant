import { useEffect, useRef } from "react";

// Confirmation dialog. Focus defaults to "Cancel" to prevent accidental confirmation; press Esc or click the backdrop to close.
export default function ConfirmDialog({ title, message, confirmLabel, cancelLabel = "Cancel", danger = false, onConfirm, onCancel }) {
  const cancelRef = useRef(null);

  useEffect(() => {
    const previous = document.activeElement;
    cancelRef.current?.focus();
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [onCancel]);

  return (
    <div className="dialog-scrim" onClick={onCancel}>
      <div
        className="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="dialog-title"
        aria-describedby="dialog-desc"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="dialog-title" className="dialog-title">
          {title}
        </h2>
        <p id="dialog-desc" className="dialog-desc">
          {message}
        </p>
        <div className="dialog-actions">
          <button type="button" className="btn" ref={cancelRef} onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className={`btn ${danger ? "btn--danger-solid" : "btn--primary"}`} onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
