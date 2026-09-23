"""
FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health          -> liveness check + which LLM backend is active
    POST /run             -> run the Researcher -> Writer pipeline on 2-3 documents
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline import ResearchWriterPipeline
from app.schemas import PipelineResult, SourceDocument

app = FastAPI(
    title="Researcher-Writer Multi-Agent API",
    description="A minimal 2-agent (Researcher -> Writer) pipeline over 2-3 source documents.",
    version="0.1.0",
)


class RunRequest(BaseModel):
    query: str = Field(..., description="The question/task for the pipeline to answer")
    documents: list[SourceDocument] = Field(
        ..., min_length=2, max_length=3, description="2 to 3 source documents"
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_mode": "real",
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
    }


@app.post("/run", response_model=PipelineResult)
def run_pipeline(payload: RunRequest) -> PipelineResult:
    try:
        pipeline = ResearchWriterPipeline()
        return pipeline.run(query=payload.query, documents=payload.documents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
