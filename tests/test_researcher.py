from app.agents.researcher import ResearcherAgent, _parse_bullets, _render_documents_block
from tests.fakes import FakeLLMClient
from app.schemas import ResearchNotes, SourceDocument


def test_render_documents_block_includes_all_docs(sample_documents):
    block = _render_documents_block(sample_documents)
    assert "id=doc1" in block
    assert "id=doc2" in block
    assert "Nghỉ phép phải gửi đơn" in block
    assert "Thưởng lương tháng 13" in block


def test_parse_bullets_extracts_source_tags():
    raw = "[doc1] Giờ làm việc 8h-12h, 13h30-17h30.\n[doc2] Thưởng tháng 13 theo công thức.\n"
    bullets = _parse_bullets(raw)
    assert len(bullets) == 2
    assert bullets[0].source_doc_id == "doc1"
    assert bullets[1].source_doc_id == "doc2"
    assert "Giờ làm việc" in bullets[0].text


def test_parse_bullets_handles_missing_tag_gracefully():
    raw = "Một dòng không có nhãn nguồn."
    bullets = _parse_bullets(raw)
    assert len(bullets) == 1
    assert bullets[0].source_doc_id == "unknown"


def test_researcher_agent_returns_research_notes(sample_documents):
    agent = ResearcherAgent(FakeLLMClient())
    notes = agent.run(query="Giờ làm việc và thưởng tháng 13 quy định thế nào?", documents=sample_documents)

    assert isinstance(notes, ResearchNotes)
    assert notes.query.startswith("Giờ làm việc")
    assert len(notes.bullets) > 0
    # Every bullet must be traceable to one of the input documents.
    valid_ids = {d.doc_id for d in sample_documents}
    for b in notes.bullets:
        assert b.source_doc_id in valid_ids


def test_researcher_agent_requires_at_least_one_document():
    agent = ResearcherAgent(FakeLLMClient())
    try:
        agent.run(query="x", documents=[])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_researcher_does_not_fabricate_bullets_absent_from_documents(sample_documents):
    """
    Sanity check on the fake test backend: bullets returned must be substrings
    (after light stripping) of the source documents they claim to come from —
    i.e. the fake is purely extractive, never invents content. This mirrors
    the "no fabrication" rule we require from a real LLM too.
    """
    agent = ResearcherAgent(FakeLLMClient())
    notes = agent.run(query="test", documents=sample_documents)
    docs_by_id = {d.doc_id: d.content for d in sample_documents}
    for b in notes.bullets:
        if b.source_doc_id in docs_by_id:
            assert b.text.strip("-+*• ") in docs_by_id[b.source_doc_id]
