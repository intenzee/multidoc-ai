"""
backend/api/chat.py

    POST   /api/retrieve                 retrieval only (no LLM, no quota) - debugging/eval
    POST   /api/chat                     ask a question, get a grounded answer with citations
    GET    /api/chat/sessions            list chat sessions
    GET    /api/chat/history?session_id= messages of one session
    DELETE /api/chat/sessions/{id}       delete a session
"""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from backend.config import get_settings
from backend.generation.answerer import answer_question
from backend.retrieval.retriever import RetrievalOptions, retrieve
from backend.storage import database as db

router = APIRouter(prefix="/api", tags=["chat"])


class OptionsIn(BaseModel):
    """Per-request retrieval settings. Omitted fields use the .env defaults."""

    mode: Literal["dense", "hybrid"] | None = None
    use_reranker: bool | None = None
    multimodal: bool | None = None
    top_k_text: int | None = Field(None, ge=1, le=30)
    top_k_images: int | None = Field(None, ge=0, le=10)
    final_top_k: int | None = Field(None, ge=1, le=15)

    def to_options(self) -> RetrievalOptions:
        return RetrievalOptions.from_settings(**self.model_dump())


class QuestionIn(BaseModel):
    question: str = Field(..., max_length=2000)
    document_ids: list[str] | None = Field(None, max_length=100)
    options: OptionsIn = Field(default_factory=OptionsIn)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("The question is empty.")
        return value


class ChatIn(QuestionIn):
    session_id: str | None = None


def _check_documents(document_ids: list[str] | None) -> list[str] | None:
    """Explicit selection must exist and be ready; no selection = all ready documents."""
    ready = [d for d in db.list_documents() if d["status"] == "ready"]
    if not ready:
        raise HTTPException(status_code=400,
                            detail="No processed documents yet. Upload a PDF and wait until "
                                   "it is ready.")
    if not document_ids:
        return None
    ready_ids = {d["id"] for d in ready}
    unknown = [i for i in document_ids if i not in ready_ids]
    if unknown:
        raise HTTPException(status_code=400,
                            detail="Some selected documents do not exist or are not ready yet.")
    return document_ids


@router.post("/retrieve")
async def retrieve_only(body: QuestionIn):
    ids = _check_documents(body.document_ids)
    result = await run_in_threadpool(retrieve, body.question, ids, body.options.to_options())
    return result.to_dict()


@router.post("/chat")
async def chat(body: ChatIn):
    ids = _check_documents(body.document_ids)
    session_id = body.session_id
    if session_id and not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if not session_id:
        session_id = db.create_session(body.question[:80])

    turns = 2 * get_settings().history_turns  # each turn = question + answer
    history = [(m["role"], m["content"]) for m in db.get_messages(session_id, turns)]
    result = await run_in_threadpool(answer_question, body.question, ids,
                                     body.options.to_options(), history)
    db.add_message(session_id, "user", body.question, {"document_ids": ids})
    db.add_message(session_id, "assistant", result["answer"], result)
    return {"session_id": session_id, **result}


@router.get("/chat/sessions")
def list_sessions():
    return db.list_sessions()


@router.get("/chat/history")
def chat_history(session_id: Annotated[str, Query()]):
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"session_id": session_id, "messages": db.get_messages(session_id)}


@router.delete("/chat/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")
