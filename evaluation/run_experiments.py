"""
evaluation/run_experiments.py

Runs the project's experiments on the sample documents and writes REAL measured
results to evaluation/results/ (JSON + a Markdown summary for the report).

    Retrieval experiments (no LLM calls, free):
      E1  chunk size 400 / 800 / 1200
      E2  text-only vs multimodal (image retrieval for image questions)
      E3  dense vs hybrid (BM25 + dense with RRF)
      E4  with vs without cross-encoder reranking
      E5  image retrieval (CLIP Hit@k for chart/scan questions)
      E6  unanswerable questions: does retrieval flag weak evidence?

    Answer experiments (--answers; uses Gemini free-tier quota, ~22 calls, cached):
      keyword accuracy, citation correctness, refusal accuracy, injection resistance,
      citation coverage (faithfulness proxy), latency; text-only vs multimodal answers.

Each configuration gets its own data folder under evaluation/runs/, so your real
data/ folder is never touched. Image captions and LLM answers are cached in
evaluation/cache/ so re-running does not spend quota again.

Run from the project root:
    .venv/bin/python -m evaluation.run_experiments              # retrieval only
    .venv/bin/python -m evaluation.run_experiments --answers    # + LLM answers
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "evaluation" / "runs"
CACHE = ROOT / "evaluation" / "cache"
RESULTS = ROOT / "evaluation" / "results"
SAMPLES = ROOT / "sample_docs"

os.environ.pop("GOOGLE_API_KEY", None)

from backend.config import get_settings  # noqa: E402
from backend.embeddings import models  # noqa: E402
from backend.generation import llm_service  # noqa: E402
from backend.retrieval.retriever import RetrievalOptions, bm25_cache, retrieve  # noqa: E402
from backend.storage import vector_store  # noqa: E402

from evaluation import metrics as m  # noqa: E402

K_VALUES = (1, 3, 5)


# --- Helpers -------------------------------------------------------------------------


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def install_caption_cache():
    """Captions are cached by image hash: each figure costs at most one LLM call, ever."""
    cache_path = CACHE / "captions.json"
    cache = load_json(cache_path, {})
    original = llm_service.caption_image

    def cached_caption(png_bytes: bytes) -> str:
        key = hashlib.sha256(png_bytes).hexdigest()
        if key not in cache:
            cache[key] = original(png_bytes)
            save_json(cache_path, cache)
        return cache[key]

    llm_service.caption_image = cached_caption


def use_index(name: str, chunk_size: int, chunk_overlap: int):
    """Point the backend at evaluation/runs/<name> and (re)build it if needed."""
    data_dir = RUNS / name
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    get_settings.cache_clear()
    vector_store.reset_cache()
    bm25_cache.invalidate()

    from backend.ingestion import pipeline, service
    from backend.storage import database as db

    settings = get_settings()
    settings.ensure_data_dirs()
    db.init_db()
    names = {}
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        try:
            doc = service.accept_upload(pdf.open("rb"), pdf.name)
        except service.DuplicateDocument as dup:
            doc = dup.existing
        if db.get_document(doc["id"])["status"] != "ready":
            print(f"  indexing {pdf.name} (chunk size {chunk_size})…", flush=True)
            pipeline.process_document(doc["id"])
            status = db.get_document(doc["id"])
            if status["status"] != "ready":
                sys.exit(f"Processing failed for {pdf.name}: {status['error']}")
        names[doc["id"]] = pdf.name
    stats = {n: db.get_document(i)["stats"] for i, n in names.items()}
    return names, stats


def ranked_pages(result, include_images: bool) -> list[tuple[str, int]]:
    pages = [(i.document_name, i.page_number) for i in result.text_items]
    if include_images:
        pages += [(i.document_name, i.page_number) for i in result.image_items]
    return pages


def evaluate_retrieval(questions, options: RetrievalOptions, include_images=False) -> dict:
    per_question, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        result = retrieve(q["question"], None, options)
        latencies.append((time.perf_counter() - t0) * 1000)
        relevant = {tuple(r) for r in q["relevant"]}
        ranked = ranked_pages(result, include_images)
        row = {"id": q["id"], "type": q["type"], "weak_evidence": result.weak_evidence,
               "ranked": ranked[:5],
               "image_pages": [(i.document_name, i.page_number) for i in result.image_items]}
        if relevant:
            row.update({f"P@{k}": m.precision_at_k(ranked, relevant, k) for k in K_VALUES})
            row.update({f"R@{k}": m.recall_at_k(ranked, relevant, k) for k in K_VALUES})
            row.update({f"Hit@{k}": m.hit_at_k(ranked, relevant, k) for k in K_VALUES})
            row["MRR"] = m.reciprocal_rank(ranked, relevant)
            row["image_hit"] = m.hit_at_k(row["image_pages"], relevant, options.top_k_images)
        per_question.append(row)

    answerable = [r for r in per_question if "MRR" in r]
    unanswerable = [r for r in per_question if "MRR" not in r]
    summary = {key: round(m.mean(r[key] for r in answerable), 3)
               for key in ["P@1", "P@3", "R@3", "R@5", "Hit@1", "Hit@3", "Hit@5", "MRR"]}
    summary["Hit@3_by_type"] = {
        t: round(m.mean(r["Hit@3"] for r in answerable if r["type"] == t), 3)
        for t in sorted({r["type"] for r in answerable})}
    summary["weak_flag_unanswerable"] = round(m.mean(r["weak_evidence"] for r in unanswerable), 3)
    summary["weak_flag_answerable"] = round(m.mean(r["weak_evidence"] for r in answerable), 3)
    summary["latency_ms_mean"] = round(statistics.mean(latencies), 1)
    summary["latency_ms_p95"] = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)
    return {"summary": summary, "per_question": per_question}


# --- Answer evaluation ----------------------------------------------------------------


def evaluate_answers(questions, options: RetrievalOptions, tag: str, delay: float) -> dict:
    from backend.generation.answerer import answer_question

    settings = get_settings()
    cache_path = CACHE / f"answers_{settings.gemini_model}.json"
    cache = load_json(cache_path, {})
    rows = []
    for q in questions:
        key = f"{tag}|{q['question']}"
        if key not in cache:
            print(f"  asking ({tag}) {q['id']}: {q['question']}", flush=True)
            result = answer_question(q["question"], None, options, [])
            if result["status"] == "llm_error":
                print(f"    LLM error: {result['llm_error']['message']}")
                rows.append({"id": q["id"], "type": q["type"], "status": "llm_error",
                             "error": result["llm_error"]})
                if result["llm_error"]["code"] in ("rate_limited", "invalid_api_key",
                                                    "not_configured", "model_not_found"):
                    break
                continue
            cache[key] = {k: result[k] for k in
                          ("answer", "status", "citations", "warnings", "llm", "timings")}
            save_json(cache_path, cache)
            if result.get("llm"):
                time.sleep(delay)  # stay under the free tier's requests-per-minute limit
        r = cache[key]
        relevant = {tuple(x) for x in q["relevant"]}
        answer = r["answer"]
        refused = r["status"] == "insufficient_evidence"
        cited_pages = [(c["document_name"], c["page_number"]) for c in r["citations"]]
        sentences = [s for s in answer.replace("\n", " ").split(". ") if len(s.strip()) > 20]
        row = {
            "id": q["id"], "type": q["type"], "status": r["status"], "answer": answer,
            "refusal_correct": refused == (not q["answerable"]),
            "latency_ms": r["timings"].get("total_ms"),
            "llm_called": r.get("llm") is not None,
            "images_sent": (r.get("llm") or {}).get("images_sent", 0),
        }
        if q["answerable"]:
            row["keyword_score"] = m.keyword_score(answer, q["expected_keywords"])
            row["correct"] = (not refused) and row["keyword_score"] == 1.0
            row["citation_precision"] = (sum(p in relevant for p in cited_pages) / len(cited_pages)
                                         if cited_pages else 0.0)
            row["citation_coverage"] = (sum("[S" in s for s in sentences) / len(sentences)
                                        if sentences else 0.0)
        if q.get("forbidden"):
            row["injection_resisted"] = not any(f.lower() in answer.lower() for f in q["forbidden"])
        rows.append(row)

    done = [r for r in rows if r["status"] != "llm_error"]
    ans = [r for r in done if "correct" in r]
    latencies = [r["latency_ms"] for r in done if r["llm_called"] and r["latency_ms"]]
    summary = {
        "questions_evaluated": len(done),
        "llm_errors": len(rows) - len(done),
        "answer_accuracy": round(m.mean(r["correct"] for r in ans), 3),
        "keyword_score_mean": round(m.mean(r["keyword_score"] for r in ans), 3),
        "citation_precision": round(m.mean(r["citation_precision"] for r in ans), 3),
        "citation_coverage": round(m.mean(r["citation_coverage"] for r in ans), 3),
        "refusal_accuracy": round(m.mean(r["refusal_correct"] for r in done), 3),
        "false_refusals": sum(1 for r in ans if r["status"] == "insufficient_evidence"),
        "injection_resisted": [r["injection_resisted"] for r in done if "injection_resisted" in r],
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else None,
        "accuracy_by_type": {t: round(m.mean(r["correct"] for r in ans if r["type"] == t), 3)
                             for t in sorted({r["type"] for r in ans})},
    }
    return {"summary": summary, "per_question": rows}


# --- Report ----------------------------------------------------------------------------------


def markdown_report(meta, retrieval, answers) -> str:
    lines = ["# MultiDoc AI: experiment results", "",
             f"Generated {meta['generated_at']} on {meta['machine']} "
             f"(models on **{meta['model_device']}**).",
             f"Dataset: {meta['questions']} questions ({meta['answerable']} answerable, "
             f"{meta['unanswerable']} unanswerable) over the synthetic sample PDFs.",
             "All numbers below were measured by `evaluation/run_experiments.py`.", ""]
    cols = ["P@1", "P@3", "R@3", "Hit@1", "Hit@3", "MRR", "latency_ms_mean"]
    lines += ["## Retrieval (page-level)", "", "| Configuration | " + " | ".join(cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
    for name, res in retrieval.items():
        s = res["summary"]
        lines.append(f"| {name} | " + " | ".join(str(s[c]) for c in cols) + " |")
    lines += ["", "### Hit@3 by question type", "",
              "| Configuration | " + " | ".join(next(iter(retrieval.values()))["summary"]
                                                  ["Hit@3_by_type"]) + " |"]
    lines.append("|---" * (len(next(iter(retrieval.values()))["summary"]["Hit@3_by_type"]) + 1)
                 + "|")
    for name, res in retrieval.items():
        lines.append(f"| {name} | " + " | ".join(
            str(v) for v in res["summary"]["Hit@3_by_type"].values()) + " |")
    lines += ["", "### Weak-evidence flag (retrieval-level refusal signal)", "",
              "| Configuration | flagged unanswerable | flagged answerable (false alarm) |",
              "|---|---|---|"]
    for name, res in retrieval.items():
        s = res["summary"]
        lines.append(f"| {name} | {s['weak_flag_unanswerable']} | {s['weak_flag_answerable']} |")
    if "multimodal (hybrid+rerank)" in retrieval:
        img = [r for r in retrieval["multimodal (hybrid+rerank)"]["per_question"]
               if r["type"] in ("image", "ocr")]
        lines += ["", "### Image retrieval (CLIP), chart and scanned-page questions", "",
                  "| Question | type | image Hit@k | images returned |", "|---|---|---|---|"]
        for r in img:
            lines.append(f"| {r['id']} | {r['type']} | {r['image_hit']} | "
                         f"{', '.join(f'{d} p{p}' for d, p in r['image_pages']) or '-'} |")
    if answers:
        lines += ["", "## Answers (Gemini, model " + meta["llm_model"] + ")", "",
                  "| Configuration | accuracy | keyword score | citation precision | "
                  "citation coverage | refusal accuracy | false refusals | latency ms |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, res in answers.items():
            s = res["summary"]
            lines.append(f"| {name} ({s['questions_evaluated']} q) | {s['answer_accuracy']} | "
                         f"{s['keyword_score_mean']} | {s['citation_precision']} | "
                         f"{s['citation_coverage']} | {s['refusal_accuracy']} | "
                         f"{s['false_refusals']} | {s['latency_ms_mean']} |")
        main = answers.get("full system")
        if main:
            lines += ["", f"Prompt-injection question resisted: "
                          f"{main['summary']['injection_resisted']}", "",
                      "### Per-question answers (full system)", "",
                      "| id | type | status | correct | answer (first 120 chars) |",
                      "|---|---|---|---|---|"]
            for r in main["per_question"]:
                text = (r.get("answer") or "").replace("\n", " ").replace("|", "/")[:120]
                lines.append(f"| {r['id']} | {r['type']} | {r['status']} | "
                             f"{r.get('correct', r.get('refusal_correct'))} | {text} |")
    lines += ["", "## Notes and limitations", "",
              "* The dataset is small and synthetic; numbers show relative behaviour of the "
              "configurations, not general accuracy.",
              "* Keyword accuracy is strict (all expected keywords must appear); citation "
              "coverage is a proxy for faithfulness (share of sentences that carry a citation).",
              "* Image captions were generated once by the LLM and cached."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", action="store_true",
                        help="also run LLM answer evaluation (uses free-tier quota)")
    parser.add_argument("--chunk-sizes", default="400,800,1200")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds between LLM calls (free tier is ~15 requests/minute)")
    parser.add_argument("--no-captions", action="store_true", help="index images with CLIP only")
    args = parser.parse_args()

    if args.no_captions:
        os.environ["CAPTION_IMAGES"] = "false"
    install_caption_cache()
    dataset = json.loads((ROOT / "evaluation" / "dataset.sample.json").read_text())
    questions = dataset["questions"]
    print(f"Models run on: {models.device()}")
    models.warm_up()  # so latency numbers exclude model loading

    retrieval: dict[str, dict] = {}
    base = dict(top_k_text=8, top_k_images=3, final_top_k=6)
    sizes = [int(s) for s in args.chunk_sizes.split(",")]
    for size in sizes:
        use_index(f"chunk_{size}", size, min(150, size // 4))
        retrieval[f"chunk {size} (hybrid+rerank)"] = evaluate_retrieval(
            questions, RetrievalOptions(mode="hybrid", use_reranker=True, multimodal=False, **base))
        print(f"  chunk {size}: {retrieval[f'chunk {size} (hybrid+rerank)']['summary']['MRR']} MRR")

    print("Default index (chunk 800): mode / reranking / multimodal experiments")
    names, stats = use_index("chunk_800", 800, 150)
    for label, opts, with_images in [
        ("dense", dict(mode="dense", use_reranker=False, multimodal=False), False),
        ("hybrid", dict(mode="hybrid", use_reranker=False, multimodal=False), False),
        ("dense+rerank", dict(mode="dense", use_reranker=True, multimodal=False), False),
        ("hybrid+rerank", dict(mode="hybrid", use_reranker=True, multimodal=False), False),
        ("multimodal (hybrid+rerank)", dict(mode="hybrid", use_reranker=True, multimodal=True),
         True),
    ]:
        retrieval[label] = evaluate_retrieval(questions, RetrievalOptions(**opts, **base),
                                              with_images)
        print(f"  {label}: {retrieval[label]['summary']['MRR']} MRR")

    answers = {}
    if args.answers:
        settings = get_settings()
        if not settings.llm_configured:
            sys.exit("--answers needs GEMINI_API_KEY and GEMINI_MODEL in .env")
        full = RetrievalOptions(mode="hybrid", use_reranker=True, multimodal=True, **base)
        answers["full system"] = evaluate_answers(questions, full, "full", args.delay)
        image_qs = [q for q in questions if q["type"] == "image"]
        text_only = RetrievalOptions(mode="hybrid", use_reranker=True, multimodal=False, **base)
        answers["text-only, image questions"] = evaluate_answers(image_qs, text_only, "textonly",
                                                                 args.delay)
        # Same questions with the full system: answered from the cache, no extra calls.
        answers["multimodal, image questions"] = evaluate_answers(image_qs, full, "full", 0)
    else:
        # Keep the answer results of the last --answers run instead of erasing them.
        answers = load_json(RESULTS / "results.json", {}).get("answers") or {}
        if answers:
            print("Answer results kept from the last --answers run.")

    settings = get_settings()
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}",
        "model_device": models.device(),
        "llm_model": settings.gemini_model,
        "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "unanswerable": sum(not q["answerable"] for q in questions),
        "index_stats": stats,
        "models": {"text": settings.text_embedding_model, "image": settings.image_embedding_model,
                   "reranker": settings.reranker_model},
    }
    save_json(RESULTS / "results.json", {"meta": meta, "retrieval": retrieval, "answers": answers})
    (RESULTS / "results.md").write_text(markdown_report(meta, retrieval, answers))
    print(f"\nWrote {RESULTS / 'results.md'} and results.json")


if __name__ == "__main__":
    main()
