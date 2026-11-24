import { useEffect, useState } from "react";
import { pageImageUrl, pdfUrl } from "../api.js";

export default function SourceModal({ source, onClose }) {
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Highlight the first line of a text source on the rendered page.
  const firstLine =
    source.kind === "text"
      ? source.text
          .split("\n")
          .map((l) => l.trim())
          .find((l) => l.length > 12 && !l.startsWith("|") && !l.startsWith("Table on page"))
      : null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Source"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <div>
            <span className="label">{source.label}</span> <strong>{source.document_name}</strong>
            <span className="muted"> · page {source.page_number}</span>
          </div>
          <div className="modal-actions">
            <a href={pdfUrl(source.document_id, source.page_number)} target="_blank" rel="noreferrer">
              Open PDF
            </a>
            <button onClick={onClose} aria-label="Close">
              ✕
            </button>
          </div>
        </div>
        <div className="modal-body">
          <div className="modal-text">
            {source.kind === "image" ? (
              <>
                <h3>Extracted image</h3>
                <img className="figure" src={source.image_url} alt="Retrieved figure" />
                {source.text && (
                  <p>
                    <em>Index description:</em> {source.text}
                  </p>
                )}
              </>
            ) : (
              <>
                <h3>Retrieved passage</h3>
                <pre>{source.text}</pre>
              </>
            )}
          </div>
          <div className="modal-page">
            <h3>Page {source.page_number}</h3>
            {imageFailed ? (
              <p className="note error">The page preview could not be loaded.</p>
            ) : (
              <img
                src={pageImageUrl(source.document_id, source.page_number, firstLine)}
                alt={`Page ${source.page_number}`}
                onError={() => setImageFailed(true)}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
