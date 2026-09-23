"""
Writer Agent.

Responsibility (and ONLY responsibility):
  - Take the `ResearchNotes` produced by the Researcher.
  - Compose a coherent final answer to the user's query using ONLY the
    facts contained in those notes.

CRITICAL ARCHITECTURAL CONSTRAINT:
  This agent's `run()` signature intentionally does NOT accept
  `list[SourceDocument]` — only `ResearchNotes`. This makes it
  structurally impossible (not just prompt-level, but code-level) for
  the Writer to access raw source documents in this pipeline.
"""
from __future__ import annotations

from app.llm_client import LLMClient
from app.schemas import FinalAnswer, ResearchNotes

WRITER_SYSTEM_PROMPT = """\
Bạn là WRITER AGENT trong một hệ thống multi-agent.

NHIỆM VỤ DUY NHẤT của bạn:
- Dựa vào RESEARCH NOTES được cung cấp bên dưới (do Researcher Agent tổng hợp),
  hãy viết một câu trả lời cuối cùng mạch lạc, chuyên nghiệp, dễ hiểu cho câu hỏi
  của người dùng.

QUY TẮC BẮT BUỘC (RẤT QUAN TRỌNG):
1. Bạn KHÔNG có quyền truy cập tài liệu nguồn gốc. Bạn CHỈ được sử dụng thông tin
   xuất hiện trong RESEARCH NOTES bên dưới.
2. TUYỆT ĐỐI KHÔNG được tự bổ sung bất kỳ thông tin thực tế (factual information)
   nào ngoài RESEARCH NOTES — không thêm số liệu, không thêm điều khoản, không suy
   đoán, không dùng kiến thức nền bên ngoài, kể cả khi bạn "biết" hoặc "đoán" được.
3. Nếu RESEARCH NOTES không đủ thông tin để trả lời một phần nào đó của câu hỏi,
   hãy nói rõ ràng là thông tin đó không có trong ghi chú, thay vì suy đoán hoặc bịa ra.
4. Viết câu trả lời bằng tiếng Việt, văn phong rõ ràng, có cấu trúc (có thể dùng gạch
   đầu dòng khi phù hợp), phù hợp để trình bày cho người dùng cuối.
5. Không cần liệt kê lại toàn bộ bullet thô — hãy tổng hợp, sắp xếp lại theo chủ đề
   sao cho dễ đọc, nhưng không được thay đổi ý nghĩa hay số liệu gốc trong notes.
"""

WRITER_USER_PROMPT_TEMPLATE = """\
CÂU HỎI / YÊU CẦU CỦA NGƯỜI DÙNG:
{query}

=== RESEARCH NOTES ===
{notes_block}
=== END RESEARCH NOTES ===

Hãy viết câu trả lời cuối cùng theo đúng quy tắc đã nêu trong system prompt.
"""


class WriterAgent:
    """Wraps an LLMClient to perform the Writer role. Sees ResearchNotes only."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def run(self, notes: ResearchNotes) -> FinalAnswer:
        # Note: signature deliberately takes only `ResearchNotes`, never
        # `list[SourceDocument]` — see module docstring.
        user_prompt = WRITER_USER_PROMPT_TEMPLATE.format(
            query=notes.query,
            notes_block=notes.as_prompt_block(),
        )
        answer_text = self._llm.chat(WRITER_SYSTEM_PROMPT, user_prompt)
        used_sources = sorted({b.source_doc_id for b in notes.bullets})
        return FinalAnswer(query=notes.query, answer=answer_text, used_sources=used_sources)
