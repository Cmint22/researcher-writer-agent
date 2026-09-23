from app.verifier import find_ungrounded_numeric_claims


def test_no_warnings_when_all_numbers_are_grounded():
    notes = "1. [doc1] Phạt lần đầu 50.000đ/lần, các lần sau 100.000đ/lần.\n2. [doc2] Nâng lương 5% đến 20%."
    answer = "Mức phạt lần đầu là 50.000đ, các lần tiếp theo 100.000đ. Nâng lương dao động 5% đến 20%."
    warnings = find_ungrounded_numeric_claims(answer, notes)
    assert warnings == []


def test_flags_a_fabricated_number_absent_from_notes():
    notes = "1. [doc1] Nghỉ phép: 1 ngày phép/tháng."
    # The Writer invents a "15 ngày" detail that was never in the notes.
    answer = "Người lao động được nghỉ phép 1 ngày/tháng và tối đa 15 ngày liên tiếp mỗi năm."
    warnings = find_ungrounded_numeric_claims(answer, notes)
    assert any("15" in w for w in warnings)


def test_tolerates_minor_reformatting_of_the_same_number():
    # notes uses "05" (as written in the source doc), answer normalizes to "5" —
    # this should NOT be flagged, since it's the same fact, just reformatted.
    notes = "1. [doc2] Cứ đủ 05 năm làm việc thì được tăng thêm 01 ngày phép."
    answer = "Cứ đủ 5 năm làm việc, người lao động được tăng thêm 1 ngày nghỉ phép."
    warnings = find_ungrounded_numeric_claims(answer, notes)
    assert warnings == []


def test_empty_answer_or_notes_produce_no_crash():
    assert find_ungrounded_numeric_claims("", "") == []
    assert find_ungrounded_numeric_claims("không có số liệu nào", "cũng không có gì") == []
