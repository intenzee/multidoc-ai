import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import Message from "./Message.jsx";
import SourceModal from "./SourceModal.jsx";

const EXAMPLES = [
  "Summarise the main points of the documents.",
  "What does the chart show?",
  "Which values are in the table?",
];

export default function Chat({ readyDocs, selected }) {
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [openSource, setOpenSource] = useState(null);
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState({ mode: "hybrid", use_reranker: true, multimodal: true });
  const endRef = useRef(null);

  const loadSessions = () => api.sessions().then(setSessions).catch(() => {});
  useEffect(() => {
    loadSessions();
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function openSession(id) {
    if (!id) return newChat();
    try {
      const data = await api.history(id);
      setSessionId(id);
      setMessages(
        data.messages.map((m) =>
          m.role === "user" ? { role: "user", text: m.content } : { role: "assistant", ...m.payload },
        ),
      );
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }

  function newChat() {
    setSessionId(null);
    setMessages([]);
    setError("");
  }

  async function send(question) {
    const q = question.trim();
    if (!q || loading) return;
    setInput("");
    setError("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const reply = await api.chat({
        session_id: sessionId,
        question: q,
        document_ids: selected.length ? selected : null,
        options,
      });
      setSessionId(reply.session_id);
      setMessages((m) => [...m, { role: "assistant", ...reply }]);
      loadSessions();
    } catch (e) {
      setError(e.message);
      setMessages((m) => m.slice(0, -1)); // let the user edit and resend
      setInput(q);
    } finally {
      setLoading(false);
    }
  }

  const noDocs = readyDocs.length === 0;

  return (
    <div className="chat">
      <div className="chat-toolbar">
        <select
          value={sessionId || ""}
          onChange={(e) => openSession(e.target.value)}
          aria-label="Chat session"
        >
          <option value="">New conversation</option>
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.title}
            </option>
          ))}
        </select>
        <button onClick={newChat} disabled={!sessionId && messages.length === 0}>
          New chat
        </button>
        <button className="ghost" onClick={() => setShowOptions((v) => !v)}>
          {showOptions ? "Hide retrieval options" : "Retrieval options"}
        </button>
      </div>

      {showOptions && (
        <div className="options">
          <label>
            Search
            <select
              value={options.mode}
              onChange={(e) => setOptions({ ...options, mode: e.target.value })}
            >
              <option value="hybrid">Hybrid (meaning + keywords)</option>
              <option value="dense">Dense (meaning only)</option>
            </select>
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.use_reranker}
              onChange={(e) => setOptions({ ...options, use_reranker: e.target.checked })}
            />
            Cross-encoder reranking
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.multimodal}
              onChange={(e) => setOptions({ ...options, multimodal: e.target.checked })}
            />
            Include images and charts
          </label>
        </div>
      )}

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <h2>Ask questions about your documents</h2>
            <p className="muted">
              Answers are generated only from your uploaded files, with citations to the page (for audio: the part of the recording).
            </p>
            {noDocs ? (
              <p className="note">Upload a file and wait until it shows “Ready”.</p>
            ) : (
              <div className="examples">
                {EXAMPLES.map((ex) => (
                  <button key={ex} onClick={() => send(ex)}>
                    {ex}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} message={m} onOpenSource={setOpenSource} />
        ))}
        {loading && (
          <div className="msg assistant">
            <div className="bubble loading">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
              Searching the documents and writing an answer…
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {error && <div className="banner error">{error}</div>}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          placeholder={noDocs ? "Upload a document first" : "Ask a question… (Shift+Enter for a new line)"}
          disabled={noDocs || loading}
          maxLength={2000}
          rows={2}
        />
        <button type="submit" disabled={noDocs || loading || !input.trim()}>
          Ask
        </button>
      </form>

      {openSource && <SourceModal source={openSource} onClose={() => setOpenSource(null)} />}
    </div>
  );
}
