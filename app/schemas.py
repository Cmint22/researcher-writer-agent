"""
Data contracts that flow through the pipeline:

    Documents  ->  [Researcher]  ->  ResearchNotes  ->  [Writer]  ->  FinalAnswer

Keeping these as explicit Pydantic models (rather than free-form strings)
is what lets us enforce the architectural rule "Writer only sees
ResearchNotes, never the raw Documents" at the type level, and makes the
hand-off between agents testable in isolation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """One input document handed to the Researcher."""

    doc_id: str = Field(..., description="Short stable identifier, e.g. 'doc1'")
    title: str = Field(..., description="Human-readable title of the document")
    content: str = Field(..., description="Raw text content of the document")


class ResearchBullet(BaseModel):
    """A single fact extracted by the Researcher, tied to its source."""

    text: str = Field(..., description="A concise, factual bullet point")
    source_doc_id: str = Field(..., description="doc_id of the document this bullet came from")


class ResearchNotes(BaseModel):
    """
    The ONLY artifact passed from Researcher to Writer.

    The Writer must not see SourceDocument.content directly — it only ever
    receives this object. This is the contract that enforces "Writer
    không được truy cập trực tiếp tài liệu nguồn".
    """

    query: str = Field(..., description="The user's original question/task")
    bullets: list[ResearchBullet] = Field(default_factory=list)

    def as_prompt_block(self) -> str:
        """Render notes as a numbered, source-tagged block for the Writer prompt."""
        if not self.bullets:
            return "(Không có ghi chú nghiên cứu nào được trích xuất.)"
        lines = []
        for i, b in enumerate(self.bullets, start=1):
            lines.append(f"{i}. [{b.source_doc_id}] {b.text}")
        return "\n".join(lines)


class FinalAnswer(BaseModel):
    """Output of the Writer agent — the end product of the pipeline."""

    query: str
    answer: str
    used_sources: list[str] = Field(
        default_factory=list, description="doc_ids referenced by the notes used"
    )


class PipelineResult(BaseModel):
    """Full trace of one pipeline run, useful for API responses / debugging / tests."""

    query: str
    documents: list[SourceDocument]
    research_notes: ResearchNotes
    final_answer: FinalAnswer
    llm_mode: str = Field(..., description="Label for which LLM backend produced this run (e.g. 'real')")
    faithfulness_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Numeric claims (percentages, VND amounts, day/month counts, ...) found in "
            "final_answer.answer that could NOT be traced back to research_notes. A "
            "non-empty list is a signal of possible fabrication by the Writer and should "
            "be reviewed — see app/verifier.py for the heuristic and its limitations."
        ),
    )
