import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import Chat from "./components/Chat.jsx";
import DocumentList from "./components/DocumentList.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import Upload from "./components/Upload.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [backendError, setBackendError] = useState("");
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]); // [] = all ready documents
  const pollRef = useRef(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await api.listDocuments();
      setDocuments(docs);
      setBackendError("");
      // Forget selections of documents that were deleted.
      setSelected((sel) => sel.filter((id) => docs.some((d) => d.id === id)));
      return docs;
    } catch (e) {
      setBackendError(e.message);
      return [];
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setBackendError(e.message));
    refreshDocuments();
  }, [refreshDocuments]);

  // Poll while any document is still being processed.
  useEffect(() => {
    const busy = documents.some((d) => d.status === "uploaded" || d.status === "processing");
    clearInterval(pollRef.current);
    if (busy) pollRef.current = setInterval(refreshDocuments, 2000);
    return () => clearInterval(pollRef.current);
  }, [documents, refreshDocuments]);

  const readyDocs = documents.filter((d) => d.status === "ready");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>◆</span> MultiDoc AI
          <span className="subtitle">Multimodal document question answering</span>
        </div>
        <div className="health">
          {health ? (
            <span className={health.llm_configured ? "pill ok" : "pill warn"}>
              {health.llm_configured
                ? `LLM: ${health.llm_model}`
                : "LLM not configured (retrieval only)"}
            </span>
          ) : (
            !backendError && <span className="pill">Connecting…</span>
          )}
        </div>
      </header>

      {backendError && <div className="banner error">{backendError}</div>}

      <main className="layout">
        <aside className="sidebar">
          <ErrorBoundary>
            <Upload onUploaded={refreshDocuments} />
            <DocumentList
              documents={documents}
              selected={selected}
              onSelect={setSelected}
              onChanged={refreshDocuments}
            />
          </ErrorBoundary>
        </aside>
        <section className="main">
          <ErrorBoundary>
            <Chat readyDocs={readyDocs} selected={selected} />
          </ErrorBoundary>
        </section>
      </main>
    </div>
  );
}
