"""
Test-only fake LLM client.

The application code (app/llm_client.py) always talks to a real,
OpenAI-compatible endpoint — it has no built-in mock/offline mode.
This fake exists purely so the test suite can run deterministically
and offline, without needing network access or an API key.

It implements the same `LLMClient` protocol (`chat(system, user) -> str`)
and is deliberately kept only under tests/, never imported by app/.
"""
from __future__ import annotations

import re


class FakeLLMClient:
    """
    Deterministic, offline stand-in for a real LLM, used only in tests.

    Implements just enough behavior for the Researcher and Writer
    prompts used in this project:

      - Researcher calls: prompt contains "=== DOCUMENT" blocks and
        asks for bullet points -> returns simple extractive bullets.
      - Writer calls: prompt contains "=== RESEARCH NOTES" and asks
        for a final answer -> returns a synthesis built only from the
        notes it was given (never invents new facts).

    This keeps the fake honest to the same architectural contract as
    the real LLM: Researcher only sees documents, Writer only sees
    notes.
    """

    DOC_BLOCK_RE = re.compile(
        r"=== DOCUMENT id=(?P<doc_id>\S+) title=\"(?P<title>[^\"]*)\" ===\n(?P<body>.*?)(?=\n=== DOCUMENT|\Z)",
        re.DOTALL,
    )
    QUERY_RE = re.compile(
        r"CÂU HỎI / YÊU CẦU CỦA NGƯỜI DÙNG:\n(?P<query>.*?)\n\n", re.DOTALL
    )
    # Explicit start/end markers (see writer.py's prompt template) make this
    # extraction exact — no risk of swallowing trailing instruction text.
    NOTES_RE = re.compile(
        r"=== RESEARCH NOTES ===\n(?P<body>.*?)\n=== END RESEARCH NOTES ===", re.DOTALL
    )

    _STOPWORDS = {
        "và", "các", "cho", "của", "là", "về", "trong", "những", "được",
        "theo", "tại", "bao", "gồm", "áp", "dụng", "người", "lao", "động",
        "quy", "định", "chế", "độ", "này", "khi", "với", "một", "có",
    }

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if "=== DOCUMENT" in user_prompt:
            return self._fake_research(user_prompt)
        if "=== RESEARCH NOTES ===" in user_prompt:
            return self._fake_write(user_prompt)
        return "(fake-llm) Không nhận diện được loại yêu cầu."

    # -- shared helpers --------------------------------------------------------
    @classmethod
    def _extract_keywords(cls, query: str) -> set[str]:
        words = re.findall(r"[a-zA-ZÀ-ỹà-ỹ0-9]+", query.lower())
        return {w for w in words if len(w) >= 4 and w not in cls._STOPWORDS}

    @staticmethod
    def _is_substantive(line: str) -> bool:
        looks_substantive = bool(
            re.match(r"^([-+*]|\d+[.)]|[a-zđ]\))", line, re.IGNORECASE) or re.search(r"\d", line)
        )
        is_boilerplate = any(
            kw in line
            for kw in [
                "CỘNG HÒA", "Độc Lập", "GIÁM ĐỐC", "Nơi nhận", "CÔNG TY CỔ PHẦN",
                "Số:", "Hà Nội, ngày", "Căn cứ",
            ]
        )
        # Bare section headers like "Điều 3: Giải thích từ ngữ" (no body content
        # on the same line) carry little standalone information — drop them,
        # their substantive sub-bullets are captured on their own lines instead.
        is_bare_header = bool(re.match(r"^(CHƯƠNG|Điều \d+[:.]?)\s*[^:]*$", line)) and len(line) < 40
        return looks_substantive and not is_boilerplate and not is_bare_header and len(line) > 8

    # -- Researcher behavior -------------------------------------------------
    def _fake_research(self, user_prompt: str) -> str:
        qm = self.QUERY_RE.search(user_prompt)
        query = qm.group("query").strip() if qm else ""
        keywords = self._extract_keywords(query)

        candidates: list[tuple[int, str, str]] = []  # (score, doc_id, clean_line)
        for m in self.DOC_BLOCK_RE.finditer(user_prompt):
            doc_id = m.group("doc_id")
            body = m.group("body")
            for raw_line in body.splitlines():
                line = raw_line.strip()
                if not line or not self._is_substantive(line):
                    continue
                clean = line.lstrip("-+*").strip()
                score = sum(1 for kw in keywords if kw in clean.lower())
                candidates.append((score, doc_id, clean))

        # Keep every line that matches at least one query keyword; if that
        # yields too little (e.g. a very short/generic query), backfill with
        # the remaining substantive lines so the notes aren't empty.
        relevant = [c for c in candidates if c[0] > 0]
        if len(relevant) < 8:
            relevant = candidates

        bullets: list[str] = []
        seen: set[str] = set()
        # Stable sort: higher relevance first, ties keep original document order.
        for score, doc_id, clean in sorted(relevant, key=lambda c: -c[0]):
            entry = f"[{doc_id}] {clean}"
            if entry not in seen:
                seen.add(entry)
                bullets.append(entry)
        return "\n".join(bullets[:50])

    # -- Writer behavior ------------------------------------------------------
    def _fake_write(self, user_prompt: str) -> str:
        m = self.NOTES_RE.search(user_prompt)
        notes_block = m.group("body").strip() if m else ""
        if not notes_block or "Không có ghi chú" in notes_block:
            return "Không có đủ thông tin trong Research Notes để trả lời câu hỏi này."

        # notes_block lines look like "3. [doc1] some fact text" — strip the
        # leading numbering added by ResearchNotes.as_prompt_block().
        lines = []
        for raw in notes_block.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            ln = re.sub(r"^\d+\.\s*", "", ln)
            lines.append(ln)

        parts = [
            "Dựa trên các quy định nội bộ đã được tổng hợp (Research Notes), dưới đây là câu trả lời:",
            "",
        ]
        for ln in lines:
            parts.append(f"- {ln}")
        parts.append("")
        parts.append(
            "(Câu trả lời trên chỉ tổng hợp lại đúng các ý có trong Research Notes được cung cấp, "
            "không bổ sung thêm thông tin thực tế nào ngoài các ghi chú đó.)"
        )
        return "\n".join(parts)
