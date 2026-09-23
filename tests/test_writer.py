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
    assert result.used_sources == ["doc1"]
    # The answer must actually be grounded in the given bullets.
    assert "8h-12h" in result.answer or "8h" in result.answer


def test_writer_reports_insufficient_notes_instead_of_fabricating():
    empty_notes = ResearchNotes(query="Chính sách bảo hiểm y tế tư nhân là gì?", bullets=[])
    agent = WriterAgent(FakeLLMClient())
    result = agent.run(empty_notes)

    # With no notes, the writer must not invent an answer.
    assert "không có đủ thông tin" in result.answer.lower() or "không có" in result.answer.lower()
    assert result.used_sources == []


def test_writer_output_only_uses_source_ids_present_in_notes():
    notes = ResearchNotes(
        query="q",
        bullets=[ResearchBullet(text="bullet A", source_doc_id="doc2")],
    )
    agent = WriterAgent(FakeLLMClient())
    result = agent.run(notes)
    assert result.used_sources == ["doc2"]
