"""
Faithfulness verifier.

The Writer's system prompt *instructs* it not to add facts beyond the
Research Notes — but a prompt instruction is not a guarantee, especially
with a real LLM. This module adds an automated, code-level check on top
of the instruction.

Heuristic: in this domain (labor/salary regulations) almost every concrete
fact is anchored to a number. We extract every numeric *claim* from the
Writer's answer as a normalized ``(value, unit)`` pair and confirm the same
pair also appears in the Research Notes.

What is normalized (so harmless re-formatting does not raise a warning):
  - leading zeros / thousand separators:  "05" == "5",  "50.000" == "50000"
  - clock times:  "8h" == "8h00" == "08:00"   (unit = "time")
  - money:        "50.000đ" == "50000 đồng";  "8 triệu" == "8.000.000đ"

What is deliberately NOT collapsed (these used to be silently accepted):
  - different units:  "12%" is NOT grounded by "12 tháng"
  - list numbering / citations: "1." and "[doc1]" are removed from BOTH
    texts before extraction, so they can never "ground" a fabricated "1".

Matching rule: a claim with a unit must match a note claim with the SAME
unit. A bare number in the answer (no unit) is accepted if the same value
appears anywhere in the notes.

This is still a heuristic, not a proof of faithfulness:
  - It cannot catch fabricated *non-numeric* claims.
  - It can still raise false positives when the Writer rephrases a number
    in a way not listed above (e.g. "tháng thứ 13" vs "13 tháng").
  - It is a safety net, not a substitute for reviewing the answer.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.citations import strip_citations

# --- unit tables ------------------------------------------------------------
_MONEY_UNITS = r"đồng|đ|vnđ|vnd"
_SCALE_WORDS = {"nghìn": 10**3, "ngàn": 10**3, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
_OTHER_UNITS = {
    "%": "percent",
    "ngày": "day",
    "tuần": "week",
    "tháng": "month",
    "quý": "quarter",
    "năm": "year",
    "giờ": "hour",
    "phút": "minute",
    "lần": "times",
}

_NUMBER = r"\d+(?:[.,]\d+)*"
_scale_alt = "|".join(_SCALE_WORDS)
_other_alt = "|".join(re.escape(u) for u in _OTHER_UNITS if u != "%")

# One alternation, tried left to right at each position, so times are
# consumed before the generic number rule can split "8h00" into "8" + "00".
_CLAIM_RE = re.compile(
    rf"""(?<!\w)(?:
        (?P<h>\d{{1,2}})h(?P<hm>\d{{2}})?(?!\w)                # 8h, 8h00, 13h30
      | (?P<c>\d{{1,2}}):(?P<cm>\d{{2}})(?!\d)                 # 08:00
      | (?P<num>{_NUMBER})
        (?:[ \t]*(?:
              (?P<scale>{_scale_alt})(?!\w)
            | (?P<money>{_MONEY_UNITS})(?!\w)
            | (?P<pct>%)
            | (?P<unit>{_other_alt})(?!\w)
        ))?
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# "1. ", "2) ", "- 3. " at the start of a line are list numbering, not facts.
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-*•+][ \t]*)?\d+[.)][ \t]+", re.MULTILINE)

Claim = tuple[str, str]  # (normalized value, unit class or "")


def _clean(text: str) -> str:
    return _LIST_MARKER_RE.sub("", strip_citations(text))


def _parse_number(raw: str) -> Decimal | None:
    """Parse a Vietnamese-formatted number: '50.000' -> 50000, '1,5' -> 1.5."""
    try:
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):  # thousand separators
            return Decimal(re.sub(r"[.,]", "", raw))
        if re.fullmatch(r"\d+[.,]\d+", raw):  # single decimal separator
            return Decimal(raw.replace(",", "."))
        if raw.isdigit():
            return Decimal(raw)
        return Decimal(re.sub(r"\D", "", raw))  # odd shapes like "1.2.3"
    except InvalidOperation:
        return None


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def extract_numeric_claims(text: str) -> dict[Claim, str]:
    """Return {(value, unit): first raw token} for every numeric claim in `text`."""
    claims: dict[Claim, str] = {}
    for m in _CLAIM_RE.finditer(_clean(text)):
        raw = m.group().strip()
        if m.group("h") is not None or m.group("c") is not None:
            hh = int(m.group("h") or m.group("c"))
            mm = int(m.group("hm") or m.group("cm") or 0)
            key: Claim = (str(hh * 60 + mm), "time")
        else:
            value = _parse_number(m.group("num"))
            if value is None:
                continue
            unit = ""
            if m.group("scale"):
                value *= _SCALE_WORDS[m.group("scale").lower()]
                unit = "money"
            elif m.group("money"):
                unit = "money"
            elif m.group("pct"):
                unit = "percent"
            elif m.group("unit"):
                unit = _OTHER_UNITS[m.group("unit").lower()]
            key = (_fmt(value), unit)
        claims.setdefault(key, raw)
    return claims


def find_ungrounded_numeric_claims(answer: str, notes_text: str) -> list[str]:
    """
    Return the raw numeric tokens in `answer` that cannot be traced to `notes_text`.

    `notes_text` should be the bullet TEXT only (``ResearchNotes.plain_text()``).
    List numbering and ``[doc_id]`` citations are stripped defensively from both
    texts anyway.
    """
    answer_claims = extract_numeric_claims(answer)
    note_claims = extract_numeric_claims(notes_text)
    note_values = {value for value, _unit in note_claims}

    ungrounded: set[str] = set()
    for (value, unit), raw in answer_claims.items():
        if unit == "":
            grounded = value in note_values  # bare number: any unit in the notes will do
        else:
            grounded = (value, unit) in note_claims
        if not grounded:
            ungrounded.add(raw)
    return sorted(ungrounded)
