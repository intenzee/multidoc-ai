// Small wrapper around the backend API. Every function throws an Error whose
// message is safe to show to the user (the backend never sends stack traces).

// Where the backend lives. Empty during development (Vite forwards /api to port 8000);
// set VITE_API_BASE when the backend runs on another server.
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");
const REMOTE = API_BASE !== "";

/** Full URL for a backend path such as "/api/images/abc". */
export const apiUrl = (path) => (path && path.startsWith("/api/") ? API_BASE + path : path);

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch {
    throw new Error(
      REMOTE
        ? "The backend is offline right now (it runs on the project owner's computer). Please try again later."
        : "Cannot reach the backend. Is the FastAPI server running on port 8000?",
    );
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ") // FastAPI validation errors
      : detail || `Request failed (HTTP ${response.status}).`;
    const error = new Error(message);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/api/health"),
  listDocuments: () => request("/api/documents"),
  getDocument: (id) => request(`/api/documents/${id}`),
  deleteDocument: (id) => request(`/api/documents/${id}`, { method: "DELETE" }),
  reprocess: (id) => request(`/api/documents/${id}/reprocess`, { method: "POST" }),
  chat: (body) => request("/api/chat", json(body)),
  sessions: () => request("/api/chat/sessions"),
  history: (sessionId) => request(`/api/chat/history?session_id=${encodeURIComponent(sessionId)}`),
  deleteSession: (id) => request(`/api/chat/sessions/${id}`, { method: "DELETE" }),
};

export const pageImageUrl = (documentId, page, highlight) => {
  const base = apiUrl(`/api/documents/${documentId}/page/${page}`);
  return highlight ? `${base}?highlight=${encodeURIComponent(highlight.slice(0, 200))}` : base;
};

export const pdfUrl = (documentId, page) =>
  apiUrl(`/api/documents/${documentId}/file#page=${page}`);

// fetch() cannot report upload progress, so uploads use XMLHttpRequest.
export function uploadDocument(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", apiUrl("/api/documents/upload"));
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new Error("Upload failed: the backend is not reachable."));
    xhr.onload = () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON reply */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else {
        const error = new Error(data?.detail || `Upload failed (HTTP ${xhr.status}).`);
        error.status = xhr.status;
        error.data = data;
        reject(error);
      }
    };
    xhr.send(form);
  });
}
