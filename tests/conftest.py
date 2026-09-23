import sys
from pathlib import Path

import pytest

# Make the project root importable as `app.*` when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import SourceDocument  # noqa: E402


@pytest.fixture
def sample_documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            doc_id="doc1",
            title="Quy định làm việc",
            content=(
                "Điều 1: Giờ làm việc\n"
                "- Làm việc từ Thứ 2 đến Thứ 6, từ 8h đến 12h, chiều từ 13h30 đến 17h30.\n"
                "- Nghỉ phép phải gửi đơn qua email trước ít nhất 3 ngày làm việc.\n"
            ),
        ),
        SourceDocument(
            doc_id="doc2",
            title="Quy chế lương thưởng",
            content=(
                "Điều 13: Thưởng lương tháng 13\n"
                "Cuối năm, người lao động được thưởng tiền lương tháng 13 tính theo công thức: "
                "[tổng lương thực tế trong năm/12 tháng] * số tháng làm việc thực tế.\n"
                "Điều 7: Nâng lương tối thiểu 6 tháng/lần, mức nâng từ 5% đến 20%.\n"
            ),
        ),
    ]
