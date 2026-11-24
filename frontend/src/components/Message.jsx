import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiUrl } from "../api.js";

const TYPE_LABEL = {
  text: "Text",
  ocr: "OCR text",
  table: "Table",
  image_caption: "Figure description",
  image: "Image",
};

// "[S1][S3]" -> markdown links "#cite-S1", rendered as clickable chips below.
function linkCitations(text) {
  return (text || "").replace(/\[(S\d+)\]/g, "[$1](#cite-$1)");
}

function SourceCard({ source, onOpen }) {
  const s = source.scores || {};
  return (
    <button className="source" onClick={() => onOpen(source)}>
      <div className="source-head">
        <span className="label">{source.label}</span>
        <span className="source-doc">{source.document_name}</span>
        <span className="muted">p. {source.page_number}</span>
        <span className="tag">{TYPE_LABEL[source.content_type] || source.content_type}</span>
      </div>
      {source.kind === "image" ? (
        <img src={apiUrl(source.image_url)} alt={`Figure on page ${source.page_number}`} loading="lazy" />
      ) : (
        <p className="excerpt">{source.text.slice(0, 220)}{source.text.length > 220 ? "…" : ""}</p>
      )}
      <div className="scores">
        {s.rerank !== undefined && <span>rerank {s.rerank.toFixed(2)}</span>}
        {s.dense !== undefined && <span>dense {s.dense.toFixed(2)}</span>}
        {s.bm25 !== undefined && <span>bm25 {s.bm25.toFixed(1)}</span>}
        {s.clip !== undefined && <span>clip {s.clip.toFixed(2)}</span>}
        {s.via_caption !== undefined && <span>via caption</span>}
      </div>
    </button>
  );
}

export default function Message({ message, onOpenSource }) {
  const [showAll, setShowAll] = useState(false);

  if (message.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{message.text}</div>
      </div>
    );
  }

  const sources = message.sources || [];
  const cited = message.citations || [];
  const byLabel = Object.fromEntries(sources.map((s) => [s.label, s]));
  const shown = showAll || cited.length === 0 ? sources : cited;

  return (
    <div className="msg assistant">
      <div className="bubble">
        {message.status === "insufficient_evidence" && (
          <div className="note warn">Not enough evidence in the documents</div>
        )}
        {message.status === "llm_error" && (
          <div className="note error">{message.llm_error?.message}</div>
        )}

        <div className="answer">
          {/* react-markdown never renders raw HTML from the model (no rehype-raw). */}
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            skipHtml
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith("#cite-")) {
                  const label = href.slice(6);
                  const source = byLabel[label];
                  return (
                    <button
                      className="cite"
                      onClick={() => source && onOpenSource(source)}
                      title={source ? `${source.document_name}, page ${source.page_number}` : ""}
                    >
                      {label}
                    </button>
                  );
                }
                return (
                  <a href={href} target="_blank" rel="noreferrer noopener">
                    {children}
                  </a>
                );
              },
            }}
          >
            {linkCitations(message.answer)}
          </ReactMarkdown>
        </div>

        {message.warnings?.map((w) => (
          <div key={w} className="note warn small">
            ⚠ {w}
          </div>
        ))}

        {sources.length > 0 && (
          <div className="sources">
            <div className="sources-head">
              <strong>
                {cited.length && !showAll ? "Cited sources" : "Retrieved sources"}
              </strong>
              {cited.length > 0 && sources.length > cited.length && (
                <button className="link" onClick={() => setShowAll((v) => !v)}>
                  {showAll ? "Show cited only" : `Show all ${sources.length} retrieved`}
                </button>
              )}
            </div>
            <div className="source-grid">
              {shown.map((s) => (
                <SourceCard key={s.label} source={s} onOpen={onOpenSource} />
              ))}
            </div>
          </div>
        )}

        {message.timings && (
          <div className="meta muted small">
            {message.timings.total_ms !== undefined &&
              `${(message.timings.total_ms / 1000).toFixed(1)} s total`}
            {message.timings.retrieval_ms !== undefined &&
              ` · retrieval ${Math.round(message.timings.retrieval_ms)} ms`}
            {message.timings.llm_ms !== undefined &&
              ` · LLM ${(message.timings.llm_ms / 1000).toFixed(1)} s`}
            {message.llm?.images_sent ? ` · ${message.llm.images_sent} image(s) sent to the model` : ""}
          </div>
        )}
      </div>
    </div>
  );
}
