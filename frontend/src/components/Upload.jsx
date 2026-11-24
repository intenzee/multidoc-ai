import { useRef, useState } from "react";
import { uploadDocument } from "../api.js";

const MAX_MB = 50; // the backend enforces the real limit (MAX_UPLOAD_MB)
// The backend checks the real file type from its content; this only gives quick feedback.
const ALLOWED = [
  ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".heic", ".heif",
  ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".markdown",
  ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm", ".mp4", ".mov", ".aiff", ".aif",
];

export default function Upload({ onUploaded }) {
  const inputRef = useRef(null);
  const [progress, setProgress] = useState(null);
  const [message, setMessage] = useState(null); // {type, text}
  const [dragging, setDragging] = useState(false);

  async function handleFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    setMessage(null);
    for (const file of files) {
      if (!ALLOWED.some((ext) => file.name.toLowerCase().endsWith(ext))) {
        setMessage({ type: "error", text: `${file.name}: this file type is not supported.` });
        continue;
      }
      if (file.size > MAX_MB * 1024 * 1024) {
        setMessage({ type: "error", text: `${file.name} is larger than ${MAX_MB} MB.` });
        continue;
      }
      try {
        setProgress({ name: file.name, value: 0 });
        await uploadDocument(file, (v) => setProgress({ name: file.name, value: v }));
        setMessage({ type: "ok", text: `${file.name} uploaded. Processing… (audio and slides can take a minute)` });
      } catch (e) {
        setMessage({ type: "error", text: `${file.name}: ${e.message}` });
      } finally {
        setProgress(null);
        onUploaded();
      }
    }
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="card">
      <h2>Upload documents</h2>
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      >
        <strong>Drop files here</strong> or click to choose
        <small>PDF · Word · PowerPoint · pictures (PNG, JPG, HEIC, TIFF) · TXT/MD · audio (MP3, WAV, M4A…) · up to {MAX_MB} MB each</small>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED.join(",")}
        multiple
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      {progress && (
        <div className="progress" aria-label="Upload progress">
          <div className="progress-label">
            {progress.name} — {Math.round(progress.value * 100)}%
          </div>
          <div className="progress-bar">
            <div style={{ width: `${progress.value * 100}%` }} />
          </div>
        </div>
      )}
      {message && <div className={`note ${message.type}`}>{message.text}</div>}
    </div>
  );
}
