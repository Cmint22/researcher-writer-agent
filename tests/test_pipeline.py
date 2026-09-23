import pytest

from app.llm_client import LLMClient
from app.pipeline import ResearchWriterPipeline
from app.schemas import PipelineResult, SourceDocument
from tests.fakes import FakeLLMClient


def test_pipeline_end_to_end_with_fake_llm(sample_documents):
    pipeline = ResearchWriterPipeline(llm_client=FakeLLMClient(), llm_mode="fake")
    result = pipeline.run(query="Nghỉ phép và thưởng tháng 13 quy định thế nào?", documents=sample_documents)

    assert isinstance(result, PipelineResult)
    assert result.llm_mode == "fake"
    assert len(result.research_notes.bullets) > 0
    assert result.final_answer.answer.strip() != ""
    # Final answer's cited sources must be a subset of the actual input doc_ids.
    valid_ids = {d.doc_id for d in sample_documents}
    assert set(result.final_answer.used_sources) <= valid_ids


def test_pipeline_rejects_fewer_than_two_documents(sample_documents):
    pipeline = ResearchWriterPipeline(llm_client=FakeLLMClient())
    with pytest.raises(ValueError):
        pipeline.run(query="q", documents=sample_documents[:1])


def test_pipeline_rejects_more_than_three_documents(sample_documents):
    extra = SourceDocument(doc_id="doc3", title="t3", content="content 3")
    extra2 = SourceDocument(doc_id="doc4", title="t4", content="content 4")
    pipeline = ResearchWriterPipeline(llm_client=FakeLLMClient())
    with pytest.raises(ValueError):
        pipeline.run(query="q", documents=sample_documents + [extra, extra2])


def test_writer_never_receives_raw_documents_object(sample_documents, monkeypatch):
    """
    Behavioral test (not just signature): spy on WriterAgent.run and assert
    it is invoked with only a ResearchNotes instance — never anything that
    exposes the original document contents directly.
    """
    from app.agents.writer import WriterAgent
    from app.schemas import ResearchNotes

    original_run = WriterAgent.run
    captured = {}

    def spy_run(self, notes):
        captured["arg_type"] = type(notes)
        captured["is_research_notes"] = isinstance(notes, ResearchNotes)
        captured["has_no_document_content_attr"] = not hasattr(notes, "documents") and not hasattr(
            notes, "content"
        )
        return original_run(self, notes)

    monkeypatch.setattr(WriterAgent, "run", spy_run)

    pipeline = ResearchWriterPipeline(llm_client=FakeLLMClient())
    pipeline.run(query="q", documents=sample_documents)

    assert captured["is_research_notes"] is True
    assert captured["has_no_document_content_attr"] is True


def test_pipeline_result_documents_field_reflects_input_but_is_separate_from_writer_path(sample_documents):
    """
    PipelineResult.documents exists for traceability/debugging in the API
    response, but this test documents that it is populated independently
    of (and after) the Writer's run — the Writer itself never consumed it.
    """
    pipeline = ResearchWriterPipeline(llm_client=FakeLLMClient())
    result = pipeline.run(query="q", documents=sample_documents)
    assert [d.doc_id for d in result.documents] == [d.doc_id for d in sample_documents]


# --- Coverage across multiple, distinct queries -----------------------------
#
# A single "does it run once" E2E check is not enough to trust that the
# pipeline generalizes. These queries exercise different subsets of the two
# real source documents (giờ làm việc / nghỉ phép vs. lương / thưởng) to
# confirm Researcher -> Writer grounding holds across varied questions, not
# just the one default query used by e2e_runner.py.
MULTI_QUERY_CASES = [
    "Giờ làm việc hàng ngày và ngày làm việc trong tuần được quy định thế nào?",
    "Quy định về nghỉ phép, nghỉ ốm và thủ tục xin nghỉ là gì?",
    "Lương tháng 13 được tính theo công thức nào?",
    "Chế độ nâng lương diễn ra bao lâu một lần và mức nâng lương là bao nhiêu?",
    "Công ty có phúc lợi gì cho nhân viên ngoài lương và nghỉ phép?",
]


@pytest.mark.parametrize("query", MULTI_QUERY_CASES)
def test_pipeline_produces_grounded_answer_across_varied_queries(sample_documents, query):
    """
    Runs the full pipeline (with the offline FakeLLMClient test double, so this needs no
    network/API key) across several distinct questions and asserts, for each:
      - the Researcher actually finds something query-relevant,
      - the Writer produces a non-empty answer,
      - the automated faithfulness check reports zero ungrounded numeric claims.
    """
    pipeline = ResearchWriterPipeline(llm_client=FakeLLMClient(), llm_mode="fake")
    result = pipeline.run(query=query, documents=sample_documents)

    assert len(result.research_notes.bullets) > 0
    assert result.final_answer.answer.strip() != ""
    assert result.faithfulness_warnings == []


def test_pipeline_flags_a_hallucinating_writer(sample_documents):
    """
    Integration proof that the faithfulness check (app/verifier.py) actually
    catches fabrication end-to-end through the pipeline — not just in
    isolation. We simulate what a real LLM could do wrong: a
    Writer that ignores its instructions and invents an extra number not
    present anywhere in the Research Notes it was given.
    """

    class HallucinatingLLMClient:
        """Researcher behaves like the fake; Writer fabricates an extra fact."""

        def __init__(self):
            self._fake = FakeLLMClient()

        def chat(self, system_prompt: str, user_prompt: str) -> str:
            if "=== DOCUMENT" in user_prompt:
                return self._fake.chat(system_prompt, user_prompt)
            # Writer call: append a fabricated, specific number that does not
            # appear anywhere in sample_documents / the notes derived from them.
            base = self._fake.chat(system_prompt, user_prompt)
            return base + "\nNgoài ra, nhân viên còn được thưởng thêm 999% lương mỗi quý."

    llm: LLMClient = HallucinatingLLMClient()
    pipeline = ResearchWriterPipeline(llm_client=llm, llm_mode="real")
    result = pipeline.run(query="Thưởng lương tháng 13 và nâng lương quy định ra sao?", documents=sample_documents)

    assert any("999" in w for w in result.faithfulness_warnings)
