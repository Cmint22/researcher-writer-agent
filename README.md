# Researcher–Writer Multi-Agent (minimal reference implementation)

A minimal, self-contained **2-agent** pipeline:

```
2–3 tài liệu
     │
     ▼
 ┌───────────┐        (bullet points, mỗi bullet gắn nhãn nguồn [doc_id])
 │ Researcher│ ─────────────────────────────────────────────┐
 └───────────┘                                               │
                                                               ▼
                                                     ResearchNotes (JSON/Pydantic)
                                                               │
                                                               ▼
                                                       ┌───────────┐
                                                       │  Writer   │
                                                       └───────────┘
                                                               │
                                                               ▼
                                                        Final Answer
```

- **Researcher** đọc các tài liệu nguồn và tổng hợp thành `ResearchNotes` — danh
  sách bullet point ngắn gọn, mỗi bullet gắn nhãn `[doc_id]` cho biết nó đến từ
  tài liệu nào.
- **Writer** viết câu trả lời cuối cùng **chỉ dựa vào** `ResearchNotes`. Writer
  **không** nhận `documents` trong signature của nó (`WriterAgent.run(self, notes: ResearchNotes)`)
  — đây là ràng buộc ở **cấp code**, không chỉ ở prompt, và được test lại trong
  `tests/test_writer.py` / `tests/test_pipeline.py`.

## Tech stack

- Python 3.10+
- FastAPI (HTTP API) + Uvicorn
- Pydantic v2 (data contracts giữa các agent)
- `openai` SDK, trỏ tới **bất kỳ endpoint OpenAI-compatible nào** qua `LLM_BASE_URL`
  (OpenAI, Azure OpenAI, Groq, Together AI, OpenRouter, Ollama local, ...)
- pytest cho unit test

### LLM backend

Dự án luôn gọi một LLM thật qua endpoint OpenAI-compatible — **không có chế độ
mock/offline** trong code ứng dụng. Bạn **bắt buộc** phải cấu hình `LLM_API_KEY`
trong `.env` (cùng `LLM_BASE_URL` / `LLM_MODEL` nếu muốn đổi provider), nếu
không `get_llm_client()` sẽ raise lỗi ngay khi khởi tạo pipeline.

Mặc định dự án được cấu hình sẵn để dùng OpenRouter với model miễn phí
`liquid/lfm-2.5-2.6b:free`:

```
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=liquid/lfm-2.5-2.6b:free
```

Bộ test (`pytest`) vẫn chạy offline/deterministic, nhưng dùng một test double
riêng (`tests/fakes.py::FakeLLMClient`) — độc lập hoàn toàn với code ứng dụng
trong `app/`, chỉ tồn tại trong `tests/`.

## Project structure

```
researcher-writer-agent/
├── app/
│   ├── __init__.py
│   ├── config.py           # đọc .env, cấu hình LLM thật (bắt buộc LLM_API_KEY)
│   ├── schemas.py          # SourceDocument, ResearchNotes, FinalAnswer, PipelineResult
│   ├── llm_client.py       # OpenAICompatibleClient (client LLM duy nhất, luôn thật)
│   ├── pipeline.py         # ResearchWriterPipeline: nối Researcher -> Writer
│   ├── main.py              # FastAPI app: GET /health, POST /run
│   └── agents/
│       ├── __init__.py
│       ├── researcher.py   # ResearcherAgent + prompt
│       └── writer.py       # WriterAgent + prompt
├── documents/               # tài liệu nguồn dùng cho E2E (2 file .txt)
│   ├── doc1_quy_dinh_lam_viec.txt
│   └── doc2_quy_che_luong_thuong.txt
├── tests/
│   ├── conftest.py
│   ├── fakes.py             # FakeLLMClient: test double, chỉ dùng trong tests/
│   ├── test_researcher.py
│   ├── test_writer.py
│   └── test_pipeline.py
├── e2e_runner.py            # MỘT LỆNH DUY NHẤT để chạy E2E
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Setup (local, VS Code)

```bash
# 1. Clone / giải nén project, mở bằng VS Code
cd researcher-writer-agent

# 2. Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Tạo file .env từ template
cp .env.example .env
# Mở .env và điền LLM_API_KEY (bắt buộc — không có chế độ mock/offline).
# LLM_BASE_URL / LLM_MODEL mặc định đã trỏ tới OpenRouter + liquid/lfm-2.5-2.6b:free.
```

## Chạy test

```bash
pytest -v
```

## Chạy E2E (một lệnh duy nhất)

```bash
python e2e_runner.py
```

Script này sẽ:
1. Đọc 2 tài liệu trong `documents/*.txt` (chính là 2 file PDF bạn cung cấp,
   đã được chuyển thành text).
2. Chạy pipeline Researcher → Writer.
3. In ra **Research Notes** và **Final Answer** ra màn hình.
4. Lưu toàn bộ trace (documents, notes, final answer, chế độ LLM đang dùng)
   vào `e2e_output.json`.

Muốn đổi câu hỏi:

```bash
python e2e_runner.py --query "Quy định về nghỉ phép và nghỉ ốm cụ thể như thế nào?"
```

## Chạy như một API (tuỳ chọn)

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET  http://localhost:8000/health` — kiểm tra trạng thái, chế độ LLM đang dùng.
- `POST http://localhost:8000/run` — chạy pipeline:

```json
{
  "query": "Tóm tắt giờ làm việc và chế độ nghỉ phép",
  "documents": [
    {"doc_id": "doc1", "title": "Quy định làm việc", "content": "..."},
    {"doc_id": "doc2", "title": "Quy chế lương thưởng", "content": "..."}
  ]
}
```

Swagger UI có sẵn tại `http://localhost:8000/docs`.

## Dùng LLM thật (OpenAI-compatible)

Điền vào `.env`:

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Đổi `LLM_BASE_URL` + `LLM_MODEL` để dùng provider khác tương thích OpenAI API
(Groq, Together AI, OpenRouter, Azure OpenAI, hoặc server local như Ollama/vLLM
có expose `/v1/chat/completions`). **Không có API key nào được hard-code trong
source code** — toàn bộ đọc từ biến môi trường qua `app/config.py`.

## Ràng buộc kiến trúc quan trọng

- `WriterAgent.run(self, notes: ResearchNotes)` — **không** có tham số nào khác,
  nên về mặt code, Writer không thể nhận `documents` gốc.
- `ResearchWriterPipeline.run()` chỉ truyền `documents` cho `ResearcherAgent.run()`,
  không bao giờ truyền cho `WriterAgent.run()`.
- Prompt của Writer cũng nhắc lại ràng buộc này ở cấp instruction, nhưng ràng buộc
  *thật sự* nằm ở signature + kiểu dữ liệu (`ResearchNotes` không chứa `SourceDocument.content`),
  không phụ thuộc vào việc LLM có "nghe lời" prompt hay không.
- `tests/test_writer.py::test_writer_signature_never_accepts_source_documents` và
  `tests/test_pipeline.py::test_writer_never_receives_raw_documents_object` kiểm
  tra trực tiếp ràng buộc này.
