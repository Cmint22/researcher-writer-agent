"""
Pipeline orchestrator.

    documents (2-3 files)
          |
          v
     [Researcher]  -- sees documents, does NOT see the Writer's output
          |
          v
    ResearchNotes (bullet points)
          |
          v
       [Writer]     -- sees ONLY ResearchNotes, never `documents`
          |
          v
      FinalAnswer

This module is the single place where both agents are wired together.
Note that `documents` is passed to `ResearcherAgent.run()` only; it is
never passed to `WriterAgent.run()`, which structurally enforces the
"Writer has no direct access to source documents" requirement.
"""
from __future__ import annotations

from app.agents.researcher import ResearcherAgent
from app.agents.writer import WriterAgent
from app.llm_client import LLMClient, get_llm_client
from app.schemas import PipelineResult, SourceDocument
from app.verifier import find_ungrounded_numeric_claims


class ResearchWriterPipeline:
    def __init__(self, llm_client: LLMClient | None = None, llm_mode: str | None = None):
        self._llm = llm_client or get_llm_client()
        self.researcher = ResearcherAgent(self._llm)
        self.writer = WriterAgent(self._llm)
        # Track which backend is in play, for transparency in the output/tests.
        # Always a real, network-backed LLM client — no mock backend exists.
        self._llm_mode = llm_mode or "real"

    def run(self, query: str, documents: list[SourceDocument]) -> PipelineResult:
        if not (2 <= len(documents) <= 3):
            raise ValueError(
                f"Pipeline expects 2-3 source documents, got {len(documents)}."
            )

        # Step 1: Researcher reads documents -> research notes.
        notes = self.researcher.run(query=query, documents=documents)

        # Step 2: Writer reads ONLY the notes -> final answer.
        final_answer = self.writer.run(notes)

        # Step 3: automated faithfulness check. This is a code-level safety
        # net on top of the Writer's prompt instruction — see app/verifier.py.
        # It can catch fabricated numbers the prompt alone did not prevent.
        warnings = find_ungrounded_numeric_claims(
            answer=final_answer.answer,
            notes_text=notes.as_prompt_block(),
        )

        return PipelineResult(
            query=query,
            documents=documents,
            research_notes=notes,
            final_answer=final_answer,
            llm_mode=self._llm_mode,
            faithfulness_warnings=warnings,
        )
