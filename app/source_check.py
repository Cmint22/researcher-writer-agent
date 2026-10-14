"""
Content-based check of the Researcher's source labels.

`parse_bullets_with_warnings` only checks that a label EXISTS among the input
documents. That does not show the bullet really came from that document: the
Researcher LLM can attach a valid-looking label (e.g. [doc1]) to a fact that
only appears in doc2. This module compares each bullet's text with the cited
document using word bigram overlap (pure code, no LLM).

Heuristic limits: a heavily paraphrased bullet has low overlap with every
document and is reported as "weak support" rather than as a proven error.
"""
from __future__ import annotations

import re

from app.schemas import UNKNOWN_SOURCE, ResearchNotes, SourceDocument

_WORD_RE = re.compile(r"\w+", re.UNICODE)

WEAK_OVERLAP = 0.25  # below this against EVERY document -> "weak support"
CITED_LOW = 0.5  # cited doc overlap below this ...
BETTER_BY = 0.3  # ... and another doc is better by at least this -> "mislabelled"


def _bigrams(text: str) -> set[tuple[str, str]]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return set(zip(words, words[1:]))


def _overlap(bullet_bg: set, doc_bg: set) -> float:
    return len(bullet_bg & doc_bg) / len(bullet_bg) if bullet_bg else 1.0


def check_source_labels(notes: ResearchNotes, documents: list[SourceDocument]) -> list[str]:
    doc_bg = {d.doc_id: _bigrams(d.content) for d in documents}
    warnings: list[str] = []
    for i, b in enumerate(notes.bullets, start=1):
        bg = _bigrams(b.text)
        if len(bg) < 3:  # too short to judge
            continue
        scores = {doc_id: _overlap(bg, dbg) for doc_id, dbg in doc_bg.items()}
        best_id = max(scores, key=scores.get)
        preview = b.text[:50]
        if scores[best_id] < WEAK_OVERLAP:
            warnings.append(
                f"Bullet {i} has weak textual support in every document "
                f"(best: {best_id} {scores[best_id]:.0%}): {preview!r}"
            )
        elif b.source_doc_id != UNKNOWN_SOURCE and b.source_doc_id in scores:
            cited = scores[b.source_doc_id]
            if cited < CITED_LOW and scores[best_id] - cited >= BETTER_BY:
                warnings.append(
                    f"Bullet {i} is labelled [{b.source_doc_id}] ({cited:.0%} overlap) but its text "
                    f"matches {best_id} ({scores[best_id]:.0%}): {preview!r}"
                )
    return warnings
