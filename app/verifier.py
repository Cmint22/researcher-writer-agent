"""
Faithfulness verifier.

The Writer's system prompt *instructs* it not to add facts beyond the
Research Notes — but a prompt instruction is not a guarantee, especially
with a real LLM. This module adds an automated, code-level check on top
of the instruction, so fabrication doesn't silently slip through when
the real model is used.

Heuristic: in this domain (labor/salary regulations), almost every
concrete fact is anchored to a number — a percentage, a day count, a
VND amount, a month count. We extract every numeric token from the
Writer's final answer and confirm each one also appears (verbatim or as
a bare digit sequence) somewhere in the Research Notes the Writer was
given. Any number that appears in the answer but not in the notes is a
strong signal of fabrication and gets flagged.

This is a heuristic, not a proof of faithfulness:
  - It cannot catch fabricated *non-numeric* claims (e.g. a made-up
    condition or exception with no number attached).
  - It can produce false positives when the Writer re-formats a number
    (e.g. writes "05" as "5", or "01/01" as "ngày 1 tháng 1") — we
    normalize digits-only as a fallback match to reduce this, but it
    is not perfect.
  - It is a safety net to catch a *class* of common, easy-to-check
    fabrications (invented percentages/amounts/day counts), not a
    substitute for reviewing the answer.
"""
from __future__ import annotations

import re

# Matches a number, optionally with a Vietnamese unit suffix commonly
# used in these documents (%, đ/đồng, ngày, tháng, giờ, năm, lần).
_NUMBER_RE = re.compile(
    r"\d[\d.,]*\s*(?:%|đ(?:ồng)?(?!\w)|ngày|tháng|giờ|năm|lần)?", re.IGNORECASE
)


def _extract_numeric_claims(text: str) -> set[str]:
    claims = set()
    for m in _NUMBER_RE.finditer(text):
        token = m.group().strip()
        if re.sub(r"\D", "", token):  # keep only tokens with at least one digit
            claims.add(token)
    return claims


def find_ungrounded_numeric_claims(answer: str, notes_text: str) -> list[str]:
    """
    Return numeric tokens present in `answer` but not traceable to `notes_text`.

    A digit-only fallback comparison (stripping units/punctuation, and
    normalizing away leading zeros) is used so that minor re-formatting by
    the LLM (e.g. "05" -> "5", "50.000đ" -> "50000 đồng") does not trigger a
    false positive.
    """

    def _normalize(token: str) -> str:
        digits = re.sub(r"\D", "", token)
        if not digits:
            return ""
        # Strip leading zeros so "05" and "5" compare equal; keep at least one digit.
        return str(int(digits))

    answer_claims = _extract_numeric_claims(answer)
    notes_claims = _extract_numeric_claims(notes_text)
    notes_normalized = {_normalize(n) for n in notes_claims if _normalize(n)}

    ungrounded = []
    for claim in answer_claims:
        normalized = _normalize(claim)
        if not normalized:
            continue
        if claim in notes_claims or normalized in notes_normalized:
            continue
        ungrounded.append(claim)
    return sorted(set(ungrounded))
