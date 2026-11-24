import { useEffect, useState } from "react";
import { api } from "../api.js";

const STATUS_LABEL = {
  uploaded: "Queued",
  processing: "Processing…",
  ready: "Ready",
  failed: "Failed",
};

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function Details({ id }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api
      .getDocument(id)
      .then((d) => active && setDetail(d))
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [id]);
  if (!detail && !error) return <div className="details">Loading…</div>;
  if (error) return <div className="details note error">{error}</div>;
  const s = detail.stats || {};
  const c = detail.counts || {};
  return (
    <div className="details">
      <ul>
        <li>Pages: {detail.page_count}{s.pages_ocr ? ` (${s.pages_ocr} read with OCR)` : ""}</li>
        <li>
          Chunks: {(c.text_chunks || 0) + (c.ocr_chunks || 0)} text
          {c.table_chunks ? `, ${c.table_chunks} table` : ""}
          {c.image_caption_chunks ? `, ${c.image_caption_chunks} figure captions` : ""}
        </li>
        <li>Images indexed: {c.images || 0}</li>
        {s.timings && (
          <li>
            Processing time: {(
              (s.timings.extract_s || 0) + (s.timings.caption_s || 0) + (s.timings.embed_s || 0)
            ).toFixed(1)} s
          </li>
        )}
      </ul>
      {s.warnings?.length > 0 && (
        <ul className="warnings">
          {s.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function DocumentList({ documents, selected, onSelect, onChanged }) {
  const [openId, setOpenId] = useState(null);
  const [error, setError] = useState("");

  function toggle(id) {
    onSelect(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  }

  async function remove(doc) {
    if (!window.confirm(`Delete "${doc.filename}"? Its file, text and images will be removed.`))
      return;
    try {
      await api.deleteDocument(doc.id);
      setError("");
    } catch (e) {
      setError(e.message);
    }
    onChanged();
  }

  async function reprocess(doc) {
    try {
      await api.reprocess(doc.id);
      setError("");
    } catch (e) {
      setError(e.message);
    }
    onChanged();
  }

  const ready = documents.filter((d) => d.status === "ready");

  return (
    <div className="card">
      <h2>Documents</h2>
      {documents.length === 0 && <p className="muted">No documents yet. Upload a PDF to start.</p>}
      {ready.length > 0 && (
        <p className="muted small">
          {selected.length === 0
            ? `Searching all ${ready.length} ready document(s). Tick documents to narrow the search.`
            : `Searching ${selected.length} selected document(s).`}
          {selected.length > 0 && (
            <button className="link" onClick={() => onSelect([])}>
              Clear selection
            </button>
          )}
        </p>
      )}
      {error && <div className="note error">{error}</div>}
      <ul className="doc-list">
        {documents.map((doc) => (
          <li key={doc.id} className="doc">
            <div className="doc-row">
              <input
                type="checkbox"
                disabled={doc.status !== "ready"}
                checked={selected.includes(doc.id)}
                onChange={() => toggle(doc.id)}
                aria-label={`Use ${doc.filename} for questions`}
              />
              <button
                className="doc-name"
                title={doc.filename}
                onClick={() => setOpenId(openId === doc.id ? null : doc.id)}
              >
                {doc.filename}
              </button>
              <span className={`status ${doc.status}`}>{STATUS_LABEL[doc.status]}</span>
            </div>
            <div className="doc-meta">
              {formatSize(doc.size_bytes)}
              {doc.page_count ? ` · ${doc.page_count} pages` : ""}
              <span className="doc-actions">
                {doc.status === "failed" && (
                  <button className="link" onClick={() => reprocess(doc)}>
                    Retry
                  </button>
                )}
                {doc.status !== "processing" && (
                  <button className="link danger" onClick={() => remove(doc)}>
                    Delete
                  </button>
                )}
              </span>
            </div>
            {doc.status === "failed" && doc.error && <div className="note error">{doc.error}</div>}
            {openId === doc.id && doc.status === "ready" && <Details id={doc.id} />}
          </li>
        ))}
      </ul>
    </div>
  );
}
