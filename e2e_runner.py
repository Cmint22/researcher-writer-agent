#!/usr/bin/env python3
"""
Single-command End-to-End runner.

Usage:
    python e2e_runner.py                          # runs ALL demo queries below
    python e2e_runner.py --query "..."             # runs exactly ONE custom query
    python e2e_runner.py --output-dir ./my_traces  # change where JSON traces go

What it does:
    1. Loads the source documents from ./documents/*.txt (2-3 files).
    2. Builds the ResearchWriterPipeline against the real, OpenAI-compatible
       LLM configured via LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in the
       environment/.env (see app/llm_client.py). LLM_API_KEY must be set.
    3. For each query, runs Researcher -> Writer -> faithfulness check.
    4. Prints Research Notes, the Final Answer, and any faithfulness
       warnings to stdout, and saves the full trace as JSON per query
       (e2e_output.json for the first/only query, e2e_output_2.json,
       e2e_output_3.json, ... for additional ones).

By default this runs several DIFFERENT queries against the same 2
documents (not just one), to demonstrate the pipeline actually
generalizes rather than being tuned to a single question. Pass
--query to run just one (custom) question instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import settings
from app.pipeline import ResearchWriterPipeline
from app.schemas import PipelineResult, SourceDocument

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Several distinct queries, each targeting a different slice of the two
# documents, run by default so a single E2E command demonstrates
# generalization instead of one hand-picked question.
DEMO_QUERIES = [
    "Tóm tắt các quy định chính về giờ làm việc, nghỉ phép/nghỉ ốm, và các chế độ "
    "lương thưởng (bao gồm lương tháng 13, nâng lương, làm thêm giờ) áp dụng cho "
    "người lao động chính thức tại công ty.",
    "Giờ làm việc hàng ngày và quy định về đi muộn về sớm là gì?",
    "Lương tháng 13 và chế độ nâng lương được tính/áp dụng như thế nào?",
    "Người lao động được hưởng những ngày nghỉ có lương nào trong năm?",
]

# Map filename -> human-readable title used in the DOCUMENT blocks sent to the LLM.
DOC_TITLES = {
    "doc1_quy_dinh_lam_viec.txt": "Quy định làm việc",
    "doc2_quy_che_luong_thuong.txt": "Quy chế Lương, Thưởng, Chế độ cho Người lao động",
}


def load_documents() -> list[SourceDocument]:
    if not DOCUMENTS_DIR.exists():
        print(f"ERROR: documents directory not found: {DOCUMENTS_DIR}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(DOCUMENTS_DIR.glob("*.txt"))
    if not (2 <= len(txt_files) <= 3):
        print(
            f"ERROR: expected 2-3 .txt files in {DOCUMENTS_DIR}, found {len(txt_files)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    docs = []
    for i, path in enumerate(txt_files, start=1):
        content = path.read_text(encoding="utf-8")
        title = DOC_TITLES.get(path.name, path.stem)
        docs.append(SourceDocument(doc_id=f"doc{i}", title=title, content=content))
    return docs


def run_one(pipeline: ResearchWriterPipeline, query: str, documents: list[SourceDocument]) -> PipelineResult:
    print(f"Query       : {query}")
    print("-" * 80)

    result = pipeline.run(query=query, documents=documents)

    print("\n### RESEARCH NOTES (Researcher -> Writer handoff) ###\n")
    print(result.research_notes.as_prompt_block())

    print("\n" + "=" * 80)
    print("### FINAL ANSWER (Writer output) ###")
    print("=" * 80 + "\n")
    print(result.final_answer.answer)

    print("\n" + "-" * 80)
    print(f"Sources referenced: {result.final_answer.used_sources}")
    if result.faithfulness_warnings:
        print(
            "WARNING (faithfulness): numeric claims in the answer not traceable "
            f"to the Research Notes: {result.faithfulness_warnings}"
        )
        print("  (Writer may have added information beyond what Researcher provided — review manually.)")
    else:
        print("OK (faithfulness): no ungrounded numeric claims detected.")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Researcher-Writer pipeline end-to-end.")
    parser.add_argument(
        "--query", "-q", default=None,
        help="Run exactly this ONE query instead of the built-in multi-query demo set.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(Path(__file__).parent),
        help="Directory to save JSON traces into (default: project root).",
    )
    args = parser.parse_args()

    if not settings.llm_api_key:
        print(
            "ERROR: LLM_API_KEY is not set. Configure it (and LLM_BASE_URL / LLM_MODEL) "
            "in your environment or .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    documents = load_documents()
    queries = [args.query] if args.query else DEMO_QUERIES

    print("=" * 80)
    print("RESEARCHER-WRITER MULTI-AGENT — E2E RUN")
    print("=" * 80)
    print(f"LLM model   : {settings.llm_model}")
    print(f"LLM base URL: {settings.llm_base_url}")
    print(f"Documents   : {[f'{d.doc_id} -> {d.title}' for d in documents]}")
    print(f"Running {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}")
    print("-" * 80)

    pipeline = ResearchWriterPipeline()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    any_warnings = False
    for i, query in enumerate(queries, start=1):
        print(f"\n{'#' * 80}\n# QUERY {i}/{len(queries)}\n{'#' * 80}")
        result = run_one(pipeline, query, documents)
        any_warnings = any_warnings or bool(result.faithfulness_warnings)

        out_path = out_dir / ("e2e_output.json" if i == 1 else f"e2e_output_{i}.json")
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nFull trace saved to: {out_path}")

    print("\n" + "=" * 80)
    print(f"DONE — {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} run.")
    if any_warnings:
        print("At least one run had faithfulness warnings — see above / trace files.")
        sys.exit(1)
    print("All runs passed the automated faithfulness check.")


if __name__ == "__main__":
    main()
