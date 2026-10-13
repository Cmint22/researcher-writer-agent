"""
Helpers for source citations of the form ``[doc_id]``.

A citation is a bracketed token that starts with a letter and contains
only word characters / hyphens, e.g. ``[doc1]``. Bracketed free text such
as ``[tổng lương / 12 tháng]`` (a formula in the source documents) is NOT
treated as a citation.
"""
from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[([A-Za-z_][\w\-]*)\]")


def extract_citations(text: str) -> list[str]:
    """Return the distinct citation ids found in ``text``, in order of first appearance."""
    seen: list[str] = []
    for m in CITATION_RE.finditer(text):
        cid = m.group(1)
        if cid not in seen:
            seen.append(cid)
    return seen


def strip_citations(text: str) -> str:
    """Remove ``[doc_id]`` citations from ``text``."""
    return CITATION_RE.sub("", text)
