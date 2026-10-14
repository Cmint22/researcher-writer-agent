from app.pipeline import ResearchWriterPipeline
from app.schemas import ResearchBullet, ResearchNotes, SourceDocument
from app.source_check import check_source_labels
from tests.fakes import FakeLLMClient

DOC1 = SourceDocument(
    doc_id="doc1", title="t1",
    content="Làm việc từ Thứ 2 đến Thứ 6 hàng tuần. Xin nghỉ phép phải gửi đơn trước ít nhất 3 ngày làm việc.",
)
DOC2 = SourceDocument(
    doc_id="doc2", title="t2",
    content="Mỗi năm Công ty sẽ xét nâng lương sau tối thiểu 6 tháng một lần, mức nâng từ 5% đến 20%.",
)


def _notes(*pairs):
    return ResearchNotes(query="q", bullets=[ResearchBullet(text=t, source_doc_id=d) for d, t in pairs])


def test_correctly_labelled_bullets_have_no_warning():
    notes = _notes(
        ("doc1", "Xin nghỉ phép phải gửi đơn trước ít nhất 3 ngày làm việc."),
        ("doc2", "Mỗi năm Công ty sẽ xét nâng lương sau tối thiểu 6 tháng một lần."),
    )
    assert check_source_labels(notes, [DOC1, DOC2]) == []


def test_bullet_labelled_with_the_wrong_existing_document_is_flagged():
    # The label doc1 exists, but this fact only appears in doc2.
    notes = _notes(("doc1", "Mỗi năm Công ty sẽ xét nâng lương sau tối thiểu 6 tháng một lần."))
    warnings = check_source_labels(notes, [DOC1, DOC2])
    assert len(warnings) == 1 and "labelled [doc1]" in warnings[0] and "doc2" in warnings[0]


def test_bullet_unsupported_by_any_document_is_flagged_as_weak_support():
    notes = _notes(("doc1", "Nhân viên được thưởng chuyến du lịch nước ngoài mỗi năm một lần."))
    warnings = check_source_labels(notes, [DOC1, DOC2])
    assert len(warnings) == 1 and "weak textual support" in warnings[0]


def test_very_short_bullets_are_not_judged():
    assert check_source_labels(_notes(("doc1", "Nghỉ phép")), [DOC1, DOC2]) == []


def test_pipeline_surfaces_mislabelled_bullet_in_citation_warnings(sample_documents):
    class MislabelLLM:
        def __init__(self):
            self._fake = FakeLLMClient()

        def chat(self, system_prompt, user_prompt):
            if "=== DOCUMENT" in user_prompt:
                # nâng lương text exists only in doc2 but is labelled doc1
                return "[doc1] Nâng lương tối thiểu 6 tháng/lần, mức nâng từ 5% đến 20%."
            return self._fake.chat(system_prompt, user_prompt)

    result = ResearchWriterPipeline(llm_client=MislabelLLM()).run("Nâng lương?", sample_documents)
    assert any("labelled [doc1]" in w for w in result.citation_warnings)
