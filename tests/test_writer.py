import inspect

from app.agents.writer import WriterAgent
from tests.fakes import FakeLLMClient
from app.schemas import FinalAnswer, ResearchBullet, ResearchNotes


def test_writer_signature_never_accepts_source_documents():
    """
    Architectural guarantee: WriterAgent.run() must not be able to receive
    raw SourceDocument objects at all — only ResearchNotes. This test fails
    loudly if someone widens the signature later (e.g. adds a `documents`
    kwarg), which would silently break the "Writer has no direct document
    access" requirement.
    """
    sig = inspect.signature(WriterAgent.run)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "notes"]
    # `from __future__ import annotations` makes annotations lazy strings,
    # so compare against the string form rather than a resolved type object.
    assert sig.parameters["notes"].annotation == "ResearchNotes"


def test_writer_produces_final_answer_from_notes():
    notes = ResearchNotes(
        query="Giờ làm việc là gì?",
        bullets=[
            ResearchBullet(text="Làm việc từ Thứ 2 đến Thứ 6, 8h-12h, 13h30-17h30.", source_doc_id="doc1"),
            ResearchBullet(text="Làm 2 ngày Thứ 7 mỗi tháng (tuần 2 và tuần 4).", source_doc_id="doc1"),
        ],
    )
    agent = WriterAgent(FakeLLMClient())
    result = agent.run(notes)

    assert isinstance(result, FinalAnswer)
    assert result.query == notes.query
    assert result.available_note_sources == ["doc1"]
    # The answer must actually be grounded in the given bullets.
    assert "8h-12h" in result.answer or "8h" in result.answer


def test_writer_reports_insufficient_notes_instead_of_fabricating():
    empty_notes = ResearchNotes(query="Chính sách bảo hiểm y tế tư nhân là gì?", bullets=[])
    agent = WriterAgent(FakeLLMClient())
    result = agent.run(empty_notes)

    # With no notes, the writer must not invent an answer.
    assert "không có đủ thông tin" in result.answer.lower() or "không có" in result.answer.lower()
    assert result.available_note_sources == []
    assert result.cited_sources == []


def test_writer_output_only_uses_source_ids_present_in_notes():
    notes = ResearchNotes(
        query="q",
        bullets=[ResearchBullet(text="bullet A", source_doc_id="doc2")],
    )
    agent = WriterAgent(FakeLLMClient())
    result = agent.run(notes)
    assert result.available_note_sources == ["doc2"]


# --- citations ---------------------------------------------------------------
class _ScriptedLLM:
    """Writer double returning a fixed answer text (test only)."""

    def __init__(self, text: str):
        self._text = text

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._text


def _two_source_notes() -> ResearchNotes:
    return ResearchNotes(
        query="q",
        bullets=[
            ResearchBullet(text="A", source_doc_id="doc1"),
            ResearchBullet(text="B", source_doc_id="doc2"),
        ],
    )


def test_available_note_sources_is_not_the_same_as_cited_sources():
    # The answer has no citation at all, yet both sources are "available".
    result = WriterAgent(_ScriptedLLM("Câu trả lời không có trích dẫn.")).run(_two_source_notes())
    assert result.available_note_sources == ["doc1", "doc2"]
    assert result.cited_sources == []
    assert result.invalid_citations == []


def test_valid_citations_are_parsed_from_answer_text():
    result = WriterAgent(_ScriptedLLM("Ý A [doc1]. Ý B [doc2]. Lại A [doc1].")).run(_two_source_notes())
    assert result.cited_sources == ["doc1", "doc2"]
    assert result.invalid_citations == []


def test_fake_citation_doc999_is_reported_invalid():
    result = WriterAgent(_ScriptedLLM("Ý A [doc1]. Ý lạ [doc999].")).run(_two_source_notes())
    assert result.cited_sources == ["doc1"]
    assert result.invalid_citations == ["doc999"]


def test_well_formed_citation_of_a_doc_not_in_the_notes_is_invalid():
    # "doc3" is syntactically fine but the notes only came from doc1 and doc2.
    result = WriterAgent(_ScriptedLLM("Ý [doc3].")).run(_two_source_notes())
    assert result.cited_sources == []
    assert result.invalid_citations == ["doc3"]


def test_unknown_source_is_never_available_or_valid_citation():
    notes = ResearchNotes(query="q", bullets=[ResearchBullet(text="A", source_doc_id="unknown")])
    result = WriterAgent(_ScriptedLLM("Ý A [unknown].")).run(notes)
    assert result.available_note_sources == []
    assert result.invalid_citations == ["unknown"]


def test_bracketed_formula_text_is_not_a_citation():
    result = WriterAgent(_ScriptedLLM("Công thức [tổng lương/12 tháng] x số tháng [doc1].")).run(
        _two_source_notes()
    )
    assert result.cited_sources == ["doc1"]
    assert result.invalid_citations == []
