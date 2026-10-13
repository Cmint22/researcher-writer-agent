"""
Researcher Agent.

Responsibility (and ONLY responsibility):
  - Read the raw source documents.
  - Synthesize them into a flat list of concise, factual, source-tagged
    bullet points relevant to the user's query.

The Researcher is the ONLY agent in this pipeline allowed to see
`SourceDocument.content`. Its output (`ResearchNotes`) is the sole
hand-off artifact to the Writer.
"""
from __future__ import annotations

from app.llm_client import LLMClient
from app.schemas import UNKNOWN_SOURCE, ResearchBullet, ResearchNotes, SourceDocument

RESEARCHER_SYSTEM_PROMPT = """\
Bạn là RESEARCHER AGENT trong một hệ thống multi-agent.

NHIỆM VỤ DUY NHẤT của bạn:
- Đọc các tài liệu nguồn được cung cấp bên dưới.
- Trích xuất và tổng hợp các thông tin liên quan đến câu hỏi/yêu cầu của người dùng
  thành các bullet point ngắn gọn, súc tích, CHÍNH XÁC theo nội dung tài liệu.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin có trong các tài liệu được cung cấp. TUYỆT ĐỐI KHÔNG
   suy diễn, KHÔNG bổ sung kiến thức bên ngoài, KHÔNG bịa thêm số liệu/điều khoản.
2. Mỗi bullet phải bắt đầu bằng nhãn nguồn ở dạng [doc_id] để Writer Agent (agent
   kế tiếp) biết bullet đó đến từ tài liệu nào. Ví dụ: [doc1] Giờ làm việc là 8h-12h, 13h30-17h30.
3. Ưu tiên các thông tin liên quan trực tiếp đến câu hỏi của người dùng, nhưng vẫn
   phải trung thực với nội dung gốc — không diễn giải sai lệch số liệu, mốc thời gian,
   điều kiện áp dụng.
4. Viết ngắn gọn, mỗi bullet là một ý độc lập, không lặp lại.
5. Không thêm lời dẫn, không thêm kết luận, không thêm nhận xét cá nhân — CHỈ trả về
   danh sách bullet, mỗi bullet một dòng, không đánh số.
6. Nếu tài liệu không chứa thông tin liên quan đến một phần nào đó của câu hỏi, đơn
   giản là không tạo bullet cho phần đó (không suy đoán để lấp đầy).
"""

RESEARCHER_USER_PROMPT_TEMPLATE = """\
CÂU HỎI / YÊU CẦU CỦA NGƯỜI DÙNG:
{query}

TÀI LIỆU NGUỒN:
{documents_block}

Hãy trả về danh sách bullet point tổng hợp theo đúng quy tắc đã nêu trong system prompt.
"""


def _render_documents_block(documents: list[SourceDocument]) -> str:
    blocks = []
    for doc in documents:
        blocks.append(f'=== DOCUMENT id={doc.doc_id} title="{doc.title}" ===\n{doc.content.strip()}')
    return "\n\n".join(blocks)


def parse_bullets_with_warnings(
    raw_text: str, valid_doc_ids: set[str] | None = None
) -> tuple[list[ResearchBullet], list[str]]:
    """
    Parse '[doc_id] bullet text' lines returned by the LLM.

    If `valid_doc_ids` is given, a label that is not one of them (e.g. a made-up
    "[doc999]") is NOT accepted as a citation: the bullet is kept but its source
    becomes UNKNOWN_SOURCE and a warning is recorded. Lines with no label at all
    are handled the same way. Without `valid_doc_ids`, labels are not validated.
    """
    bullets: list[ResearchBullet] = []
    warnings: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        label = None
        text = line
        if line.startswith("[") and "]" in line:
            head, _, rest = line.partition("]")
            label, text = head.lstrip("[").strip(), rest.strip()
            if not text:
                continue
        preview = text[:60]
        if label is None:
            warnings.append(f"Bullet without a source label (kept as '{UNKNOWN_SOURCE}'): {preview!r}")
            label = UNKNOWN_SOURCE
        elif valid_doc_ids is not None and label not in valid_doc_ids:
            warnings.append(
                f"Bullet labelled with unknown source [{label}] "
                f"(not in input documents; kept as '{UNKNOWN_SOURCE}'): {preview!r}"
            )
            label = UNKNOWN_SOURCE
        bullets.append(ResearchBullet(text=text, source_doc_id=label))
    return bullets, warnings


def _parse_bullets(raw_text: str, valid_doc_ids: set[str] | None = None) -> list[ResearchBullet]:
    """Backward-compatible wrapper returning only the bullets."""
    return parse_bullets_with_warnings(raw_text, valid_doc_ids)[0]


class ResearcherAgent:
    """Wraps an LLMClient to perform the Researcher role."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def run(self, query: str, documents: list[SourceDocument]) -> ResearchNotes:
        if not documents:
            raise ValueError("ResearcherAgent requires at least one document.")

        user_prompt = RESEARCHER_USER_PROMPT_TEMPLATE.format(
            query=query,
            documents_block=_render_documents_block(documents),
        )
        raw_output = self._llm.chat(RESEARCHER_SYSTEM_PROMPT, user_prompt)
        bullets, warnings = parse_bullets_with_warnings(
            raw_output, valid_doc_ids={d.doc_id for d in documents}
        )
        return ResearchNotes(query=query, bullets=bullets, source_warnings=warnings)
