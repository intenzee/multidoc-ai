# MultiDoc AI

**A Multimodal Retrieval-Augmented Generation System for Intelligent Document Question Answering**

Upload PDFs, Word and PowerPoint files, pictures (PNG, JPG, WEBP, TIFF, HEIC: charts, photos, scans), text files (.txt, .md) or audio recordings, and ask questions in plain language.
MultiDoc AI finds the relevant passages *and images*, sends them to an LLM, and returns an
answer that cites the exact document and page. If the documents don't contain the answer,
it says so instead of guessing.

B.Tech AIML minor project.

---

## Features

| Area | What it does |
|---|---|
| Ingestion | Every file becomes a PDF first: pictures (one page each), Word/PowerPoint (LibreOffice, headless), text, and audio (transcribed locally with Whisper on the Apple GPU, one page per 2 minutes, with timestamps). Then: PDF text (PyMuPDF), tables to Markdown (pdfplumber), embedded images, OCR for scanned pages (Tesseract), page renders for vector charts, header/footer and duplicate-page removal |
| Validation | PDF magic-byte check, size and page limits, corrupted/encrypted/empty PDFs rejected with clear messages, SHA-256 duplicate detection, safe file names (no path traversal) |
| Indexing | MiniLM text embeddings (384-d) + CLIP image embeddings (512-d) in separate ChromaDB collections; facts in SQLite |
| Retrieval | Dense search, BM25 keyword search, Reciprocal Rank Fusion, cross-encoder reranking, CLIP text-to-image search, weak-evidence detection, page-spread evidence for broad questions (summaries, lists of topics) |
| Generation | Gemini receives the evidence text **and the original images** (as PNG bytes), answers with `[S1]`-style citations; invalid citations are removed; refusal marker for unanswerable questions; prompt-injection defences |
| Chat | Sessions with history; short follow-up questions reuse the previous question for retrieval |
| Frontend | React + Vite: upload with progress, document status and statistics, document selection, chat, citation chips, source viewer with highlighted page render |
| Evaluation | P@k, R@k, MRR, Hit@k, answer accuracy, citation precision, refusal accuracy, latency; 6 experiments |

## Architecture

```
React (Vite, :5173) ──/api proxy──► FastAPI (:8000)
                                     │
   upload ─► validate ─► SQLite ─► background processing:
                                     extract text / OCR / tables / images
                                     chunk ─► caption figures (LLM, limited)
                                     embed (MiniLM, CLIP; GPU if available)
                                     store ─► ChromaDB + SQLite
                                     │
   question ─► dense + BM25 ─► RRF ─► cross-encoder rerank ─┐
           └─► CLIP text→image search ───────────────────────┤
                                                             ▼
                          grounded prompt (evidence [S1..Sn] + images)
                                     ─► Gemini ─► citation check ─► answer
```

| Folder | Contents |
|---|---|
| `backend/api/` | HTTP endpoints (documents, chat, retrieval) |
| `backend/ingestion/` | upload validation, PDF extraction, chunking, processing pipeline |
| `backend/embeddings/` | the three local models (text, image, reranker) |
| `backend/retrieval/` | dense, BM25, fusion, reranking, image retrieval |
| `backend/generation/` | LLM provider abstraction, prompt, grounded answers |
| `backend/storage/` | SQLite and ChromaDB access |
| `frontend/` | React user interface |
| `evaluation/` | dataset, metrics, experiment runner, results |
| `scripts/` | sample-document generator, Gemini connection checks |
| `tests/` | pytest suite (no LLM calls) |

## Setup (macOS)

Requirements: Python 3.12, Node.js 20+, Tesseract. Optional: LibreOffice (Word/PowerPoint) and ffmpeg (audio).

```bash
brew install python@3.12 tesseract node ffmpeg
brew install --cask libreoffice   # only needed for Word and PowerPoint files
```

```bash
cd multidoc-ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from the template and add your own key from Google AI Studio
(it stays on your machine and is ignored by git):

```bash
cp .env.example .env
```

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Check the connection (one tiny free-tier request each):

```bash
python scripts/check_gemini.py
python scripts/check_gemini_vision.py
```

Install the frontend:

```bash
cd frontend && npm install
```

## Run

Terminal 1, backend:

```bash
.venv/bin/uvicorn backend.main:app --reload --reload-dir backend
```

Terminal 2, frontend:

```bash
cd frontend && npm run dev
```

Open http://localhost:5173. API documentation: http://127.0.0.1:8000/docs.

The first upload downloads the local models (about 700 MB in total) into `~/.cache/huggingface`.

## GPU

The local models (MiniLM, CLIP, cross-encoder) run on the GPU automatically when one is available:
Apple Silicon (`mps`) or NVIDIA (`cuda`), otherwise the CPU. Override in `.env`:

```
MODEL_DEVICE=auto   # auto | mps | cuda | cpu
```

No model is trained in this project: all three are pre-trained and used for inference only.
The device used is recorded in each document's statistics and in the evaluation results.

## Configuration

All settings live in `backend/config.py` and can be overridden in `.env`, for example:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` / `MAX_PAGES` | 50 / 300 | upload limits |
| `WHISPER_MODEL` | mlx-community/whisper-small-mlx | local speech-to-text model for audio |
| `MAX_AUDIO_MINUTES` / `TRANSCRIPT_PAGE_SECONDS` | 60 / 120 | audio length limit / seconds of audio per "page" |
| `OFFICE_TIMEOUT_S` | 180 | time limit for converting a Word/PowerPoint file |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | characters per chunk |
| `RETRIEVAL_MODE` | hybrid | `dense` or `hybrid` |
| `USE_RERANKER` / `MULTIMODAL` | true / true | retrieval features |
| `CAPTION_IMAGES` / `MAX_CAPTIONS_PER_DOCUMENT` | true / 10 | LLM figure descriptions at indexing time (uses quota) |
| `RERANK_MIN_SCORE` | 0.0 | below this, retrieval is flagged as weak and figures are not attached via captions (the LLM still decides whether it can answer) |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, LLM configured (never calls the LLM) |
| POST | `/api/documents/upload` | upload a PDF or a PNG/JPG/WEBP picture (201, or 400/409/413/415/422) |
| GET | `/api/documents` | list with status |
| GET / DELETE | `/api/documents/{id}` | details and statistics / delete everything |
| POST | `/api/documents/{id}/reprocess` | process again |
| GET | `/api/documents/{id}/file` | original PDF |
| GET | `/api/documents/{id}/page/{n}?highlight=` | page as PNG, optional highlight |
| GET | `/api/images/{image_id}` | extracted image |
| POST | `/api/retrieve` | retrieval only, no LLM |
| POST | `/api/chat` | grounded answer with citations |
| GET | `/api/chat/sessions`, `/api/chat/history?session_id=` | chat history |

## Tests and evaluation

```bash
.venv/bin/python -m pytest -q tests
```

Tests use a temporary data folder and never call the LLM.

```bash
.venv/bin/python scripts/make_sample_docs.py
.venv/bin/python -m evaluation.run_experiments
.venv/bin/python -m evaluation.run_experiments --answers
```

The first command makes synthetic PDFs with known facts: text, a table, a chart, a scanned page,
a contradiction and a prompt-injection line. The second runs the retrieval experiments, which are free.
The third also runs the answer experiments: about 22 free-tier Gemini calls, cached in `evaluation/cache/`.
Results are written to `evaluation/results/results.md` and `results.json`.
Report only numbers from these files.

## Security

- The Gemini key is read only by the backend from `.env`; it is never sent to the browser, logged or committed.
- Document content is treated as untrusted: it is wrapped in evidence tags, the model is told never to follow
  instructions inside it, and documents cannot close their evidence block.
- LLM output is rendered as Markdown without raw HTML.
- Uploaded files are stored under random IDs; the original name is only displayed.
- The file type is checked from the content, not the name. Pictures are decoded with a pixel limit (40 MP) to block decompression bombs.
- Internal errors return a generic message; details stay in the server log.

## Limitations

- **Audio:** transcription runs locally with `mlx-whisper` (Apple-silicon Macs only; the
  first recording downloads the model, about 480 MB). Recordings are limited to
  `MAX_AUDIO_MINUTES` (60) and to the 50 MB upload limit. Citations point to the
  2-minute part of the recording; the transcript lines carry timestamps. For better
  accuracy set `WHISPER_MODEL=mlx-community/whisper-large-v3-turbo` (slower, ~1.6 GB).
- **Word/PowerPoint:** layout comes from LibreOffice, which can differ slightly from
  Microsoft Office. Spreadsheets (.xlsx) are not supported.

- OCR quality depends on scan quality; handwriting is not supported.
- Complex tables with merged cells may lose structure.
- Free-tier LLM quotas limit how many questions and figure captions can be processed per day.
- The evaluation dataset is small and synthetic.
