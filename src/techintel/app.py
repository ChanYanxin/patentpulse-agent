"""FastAPI app: launch with `uvicorn techintel.app:app --reload`."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from .demo_ui import DEMO_HTML
from pydantic import BaseModel, Field
from .retrieval import KnowledgeIndex
from .memory import SessionMemory
from .agent import TechIntelAgent
from .patents import parse_claims, compare_claims, search_local_patents, search_curated_patent_notes, search_patentsview

BASE = Path(__file__).resolve().parents[2]


def corpus_dirs():
    root = Path(os.getenv("TECHINTEL_CORPUS_DIR", str(BASE / "corpus")))
    return [root / "demo", root / "source_notes", root / "patents_demo", root / "patents_source_notes", root / "your_docs"]


@lru_cache

def get_agent() -> TechIntelAgent:
    db_path = os.getenv("TECHINTEL_MEMORY_DB", ":memory:")
    return TechIntelAgent(KnowledgeIndex(corpus_dirs()), SessionMemory(db_path))


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=3000)
    session_id: str = Field(default="demo", min_length=1, max_length=80)
    mode: Literal["offline", "llm"] = "offline"


class PatentSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    k: int = Field(default=3, ge=1, le=10)
    source: Literal["local", "curated", "patentsview"] = "local"
    retrieval: Literal["hybrid", "word", "bm25"] = "hybrid"


class ClaimsRequest(BaseModel):
    claims_text: str = Field(min_length=1, max_length=60000)


class ClaimChartRequest(BaseModel):
    target_claims: str = Field(min_length=1, max_length=60000)
    reference_claims: str = Field(min_length=1, max_length=60000)


class NoteRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)
    note: str = Field(min_length=1, max_length=1500)


app = FastAPI(title="PatentPulse-Agent", version="0.2.0",
              description="Evidence-grounded technical intelligence MVP (demo corpus is synthetic)")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    return HTMLResponse(DEMO_HTML)


@app.get("/health")
def health():
    return {"status": "ok", "documents": get_agent().index.document_count}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return get_agent().run(req.query, req.session_id, req.mode).to_dict()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # In a real service use structured logging/tracing instead of returning internals.
        raise HTTPException(status_code=503, detail="Research agent temporarily unavailable") from exc


@app.post("/memory")
def save_memory(req: NoteRequest):
    return get_agent().memory.add(req.session_id, req.note)


@app.get("/memory/{session_id}")
def read_memory(session_id: str):
    return {"session_id": session_id, "notes": get_agent().memory.recent(session_id)}


@app.get("/documents")
def documents():
    idx = get_agent().index
    return [{"doc_id": doc_id, "title": chunks[0].title} for doc_id, chunks in sorted(idx.docs.items())]


@app.post("/patents/search")
def patents_search(req: PatentSearchRequest):
    try:
        if req.source == "local":
            return search_local_patents(get_agent().index, req.query, req.k, req.retrieval)
        if req.source == "curated":
            return search_curated_patent_notes(get_agent().index, req.query, req.k, req.retrieval)
        return search_patentsview(req.query, req.k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/patents/claims/parse")
def patents_parse(req: ClaimsRequest):
    try:
        return parse_claims(req.claims_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/patents/claims/compare")
def patents_compare(req: ClaimChartRequest):
    try:
        return compare_claims(req.target_claims, req.reference_claims)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
