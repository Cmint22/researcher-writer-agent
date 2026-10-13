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


# --- regression tests for the blind spots found in review ---------------------------
import pytest  # noqa: E402


def test_fabricated_1_is_not_masked_by_bullet_number_or_doc_id():
    notes = "1. [doc1] Không có số liệu về ngày nghỉ."
    answer = "Người lao động được nghỉ 1 ngày."
    assert find_ungrounded_numeric_claims(answer, notes) == ["1 ngày"]


def test_fabricated_2_is_not_masked_by_second_bullet_and_doc2():
    notes = "1. [doc1] Làm việc Thứ 2 đến Thứ 6.\n2. [doc2] Không có thông tin khác."
    assert find_ungrounded_numeric_claims("Được nghỉ 2 ngày.", notes) == ["2 ngày"]


def test_bullet_numbering_in_the_answer_is_not_flagged():
    notes = "Giờ làm việc 8h đến 12h\nNghỉ phép trước 3 ngày"
    answer = "1. Giờ làm việc 8h đến 12h\n2. Nghỉ phép trước 3 ngày\n3) Hết"
    assert find_ungrounded_numeric_claims(answer, notes) == []


def test_citations_in_the_answer_are_not_flagged():
    assert find_ungrounded_numeric_claims("Nghỉ 3 ngày [doc1] và [doc2]", "Nghỉ 3 ngày") == []


def test_number_matching_bullet_index_but_absent_from_text_via_pipeline_style_input():
    # Same as the prompt block the old pipeline passed; numbering is stripped now.
    notes = "1. [doc1] A\n2. [doc1] B\n3. [doc2] C"
    assert find_ungrounded_numeric_claims("Có 3 ngày nghỉ.", notes) == ["3 ngày"]


def test_12_months_is_not_12_percent():
    assert find_ungrounded_numeric_claims("Mức áp dụng là 12%.", "Mức áp dụng là 12 tháng.") == ["12%"]


def test_12_percent_is_grounded_by_12_percent():
    assert find_ungrounded_numeric_claims("Mức là 12%.", "Mức là 12 %.") == []


def test_50000_dong_is_not_50000_days():
    assert find_ungrounded_numeric_claims("50.000 ngày", "Phạt 50.000đ") == ["50.000 ngày"]


@pytest.mark.parametrize(
    "answer, notes",
    [
        ("Phạt 50000 đồng", "Phạt 50.000đ/lần"),
        ("Phạt 50.000đ", "Phạt 50000 đồng"),
        ("Thưởng 8 triệu", "Thưởng 8.000.000đ"),
        ("Thưởng 8.000.000 đồng", "Thưởng 8 triệu"),
    ],
)
def test_money_formats_are_equivalent(answer, notes):
    assert find_ungrounded_numeric_claims(answer, notes) == []


@pytest.mark.parametrize(
    "answer, notes",
    [
        ("Bắt đầu lúc 8h00", "Bắt đầu lúc 8h"),
        ("Bắt đầu lúc 8h", "Bắt đầu lúc 8h00"),
        ("Bắt đầu lúc 08:00", "Bắt đầu lúc 8h"),
        ("Bắt đầu lúc 8h00", "Bắt đầu lúc 08:00"),
        ("Chiều 13h30", "Chiều 13:30"),
    ],
)
def test_time_formats_are_equivalent(answer, notes):
    assert find_ungrounded_numeric_claims(answer, notes) == []


def test_different_times_are_not_equivalent():
    assert find_ungrounded_numeric_claims("Bắt đầu lúc 9h", "Bắt đầu lúc 8h") == ["9h"]


def test_time_8h00_is_not_split_into_8_and_00():
    notes = "Từ 8h đến 12h sáng và 13h30 đến 17h30 chiều"
    assert find_ungrounded_numeric_claims("Từ 8h00 đến 12h00 và 13h30 đến 17h30", notes) == []


def test_vietnamese_word_den_is_not_swallowed_as_a_money_unit():
    # Regression for the 'đến' bug: "00 đ" must never appear as a token.
    answer = "Buổi sáng: Từ 8h00 đến 12h00\nBuổi chiều: Từ 13h30 đến 17h30"
    notes = "Thời gian làm việc cụ thể: Từ 8h đến 12h sáng và 13h30 đến 17h30 chiều"
    assert find_ungrounded_numeric_claims(answer, notes) == []
    assert find_ungrounded_numeric_claims("làm từ 5 đến 7 giờ", "làm 5 đến 7 giờ") == []


def test_trailing_sentence_period_is_not_part_of_the_number():
    assert find_ungrounded_numeric_claims("Làm đến 12h00.", "Làm đến 12h") == []
    assert find_ungrounded_numeric_claims("Phạt 15.", "Phạt 15") == []


def test_decimal_numbers():
    assert find_ungrounded_numeric_claims("Gấp 1,5 lần", "Gấp 1.5 lần") == []
    assert find_ungrounded_numeric_claims("Gấp 1,5 lần", "Gấp 15 lần") == ["1,5 lần"]


def test_doc_ids_written_without_brackets_are_not_numeric_claims():
    assert find_ungrounded_numeric_claims("Theo doc1 và doc2 thì nghỉ 3 ngày", "Nghỉ 3 ngày") == []


def test_unit_added_in_answer_is_flagged_but_bare_number_in_answer_is_accepted():
    # Documented strictness: a claim WITH a unit must match the same unit in the notes.
    assert find_ungrounded_numeric_claims("Nghỉ 7 ngày", "Làm Thứ 7") == ["7 ngày"]
    # A bare number in the answer is accepted if the value appears in the notes.
    assert find_ungrounded_numeric_claims("Làm Thứ 7", "Nghỉ 7 ngày") == []
