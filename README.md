# Humana Agent — Enterprise Agentic RAG System

An enterprise-grade, fully configurable **Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **LangChain**, **ChromaDB**, and **OpenAI**.  Uploads PDF documents through a **Streamlit** UI, extracts and indexes content at the page level, and answers questions through a three-agent pipeline with guardrails and inline citations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                            │
│   📤 Upload Tab   │   💬 Chat Tab   │   📚 Registry Tab         │
└────────┬──────────┴────────┬─────────┴──────────────────────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌──────────────────────────────────────────┐
│ Ingestion       │  │  LangGraph Agent Orchestrator            │
│ Pipeline        │  │                                          │
│                 │  │  ┌────────────┐   ┌─────────────┐        │
│ 1. PDF Extract  │  │  │ Retriever  │──▶│  Generator  │        │
│    (pdfplumber  │  │  │  Agent     │   │   Agent     │        │
│     + OCR)      │  │  └────────────┘   └──────┬──────┘        │
│ 2. JSON Persist │  │                          │               │
│ 3. Chunking     │  │                   ┌──────▼──────┐        │
│ 4. Embedding    │  │                   │  Validator  │        │
│    (3-large)    │  │                   │   Agent     │        │
│ 5. ChromaDB     │  │                   └──────┬──────┘        │
│    upsert       │  │           retry ◀────────┘ │ pass        │
└────────┬────────┘  │                            ▼             │
         │           │                           END            │
         ▼           └──────────────────────────────────────────┘
   ┌───────────┐
   │ ChromaDB  │◀──── All agents read/write through vector_store.py
   └───────────┘
```

### Agent Responsibilities

| Agent | Role | Key Config Keys |
|---|---|---|
| **Retriever** | Embeds query → semantic search → builds formatted context with source labels | `retriever.top_k`, `retriever.score_threshold` |
| **Generator** | Constructs prompt + calls LLM → produces response with inline citations | `llm.*`, `prompts.system` |
| **Validator** | Runs toxicity & relevance guardrails → approves or retries | `guardrails.*` |

---

## Project Structure

```
humana-agent/
│
├── config.yaml                    ← Single source of truth for ALL settings
├── requirements.txt
├── .env.example                   ← Copy to .env and add API keys
├── app.py                         ← Streamlit entry point
│
├── ingestion/
│   ├── pdf_extractor.py           ← pdfplumber + OCR (pytesseract) + image extraction
│   ├── chunker.py                 ← RecursiveCharacterTextSplitter with metadata
│   ├── embedder.py                ← OpenAI text-embedding-3-large (batched)
│   ├── vector_store.py            ← ChromaDB adapter with dedup logic
│   └── pipeline.py                ← Orchestrates steps 1-7 for a single PDF
│
├── agents/
│   ├── state.py                   ← AgentState TypedDict
│   ├── retriever_agent.py         ← LangGraph retriever node
│   ├── generator_agent.py         ← LangGraph generator node (multi-provider)
│   ├── validator_agent.py         ← LangGraph validator node
│   └── graph.py                   ← StateGraph definition + run_query() entrypoint
│
├── guardrails/
│   └── checker.py                 ← check_toxicity(), check_relevance()
│
├── utils/
│   ├── config_loader.py           ← YAML → ConfigNode (dot-notation access)
│   ├── hash_utils.py              ← SHA-256 helpers for deduplication
│   └── logger.py                  ← Structured JSON logger
│
└── data/                          ← Auto-created at runtime
    ├── chroma_db/                 ← Persistent vector store
    ├── extracted_json/            ← Per-document page-level JSON
    ├── ingestion_registry.json    ← File-level dedup registry
    └── logs/app.log               ← Structured JSON logs
```

---

## Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Tesseract OCR | 5.x (for scanned PDFs) |
| Poppler | latest (required by pdf2image) |

**Install Tesseract (Windows):**
Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.

**Install Poppler (Windows):**
Download from https://github.com/oschwartz10612/poppler-windows/releases and add `bin/` to PATH.

### Installation

```bash
# 1. Clone / open the project folder
cd "Humana Agent"

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Configuration Reference (`config.yaml`)

All runtime behaviour is controlled from `config.yaml`.  **No code changes are needed** to swap models, providers, or tune parameters.

### Switch LLM provider

```yaml
# OpenAI (default)
llm:
  provider: openai
  model: gpt-4o

# Azure OpenAI
llm:
  provider: azure
  azure_deployment: my-gpt4o-deployment
  azure_endpoint_env: AZURE_OPENAI_ENDPOINT

# Anthropic Claude
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
```

### Tune retrieval

```yaml
retriever:
  top_k: 5               # increase for broader context
  score_threshold: 0.30  # lower = more permissive
```

### Tune chunking

```yaml
ingestion:
  chunk_size: 1000    # characters per chunk
  chunk_overlap: 200  # overlap keeps context across chunk boundaries
```

### Tune guardrails

```yaml
guardrails:
  check_relevance: true
  relevance_threshold: 0.35   # overlap fraction required
  use_openai_moderation: true # set false to use regex-only toxicity check
  max_retries: 2              # generator retries before fallback
```

### Disable OCR (digital PDFs only)

```yaml
ingestion:
  ocr_enabled: false
```

---

## Deduplication Design

Two layers prevent duplicate content:

| Layer | Mechanism | Scope |
|---|---|---|
| **File level** | SHA-256 of raw file bytes stored in `ingestion_registry.json` | Whole document |
| **Chunk level** | SHA-256 of `filename + page + chunk_index` used as ChromaDB document ID | Individual chunk |

If you re-upload a modified version of a document (changed content → new file hash), the new version is fully re-ingested.  Identical re-uploads are silently skipped.

---

## Page-Level JSON Schema

Every ingested PDF produces a `data/extracted_json/<name>_extracted.json` file.

```json
[
  {
    "doc_id":       "sha256(filename::page::N)",
    "filename":     "policy.pdf",
    "page_number":  3,
    "total_pages":  20,
    "text":         "…extracted or OCR'd text…",
    "tables": [
      {
        "table_id": "page3_table0",
        "headers":  ["Col A", "Col B"],
        "rows":     [["val1", "val2"]],
        "row_count": 1,
        "col_count": 2
      }
    ],
    "images": [
      {
        "image_id":   "page3_img0",
        "format":     "png",
        "base64":     "iVBORw0KGgo…",
        "size_bytes": 14520
      }
    ],
    "is_scanned": false,
    "metadata": {
      "file_hash":    "sha256(file_bytes)",
      "source_path":  "/tmp/policy.pdf",
      "ingested_at":  "2026-07-22T10:30:00+00:00",
      "page_bbox":    [0, 0, 612, 792],
      "has_tables":   true,
      "has_images":   false,
      "text_length":  842,
      "ocr_applied":  false
    }
  }
]
```

---

## Citation Format

Every assistant response includes inline citations automatically:

> "The deductible resets on January 1st each year. **[Source: benefits_guide.pdf, Page 12]**"

The **Sources** panel below each response shows:
- Document filename
- Page number
- Relevance score (cosine similarity)

---

## Guardrails

| Check | Mechanism | Config Key |
|---|---|---|
| **Empty response** | Length < 10 chars | — |
| **Toxicity** | OpenAI Moderation API → regex fallback | `guardrails.use_openai_moderation` |
| **Relevance** | Keyword overlap ≥ threshold between response and context | `guardrails.relevance_threshold` |
| **Max retries** | After N failed validation rounds, return safe fallback | `guardrails.max_retries` |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (default) | OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure only | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure only | Azure OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic only | Anthropic API key |

---

## Development Notes

- All log output is **structured JSON** written to both stdout and `data/logs/app.log`.
- The LangGraph compiled graph and the ChromaDB store are **module-level singletons** — they are initialised once per process.
- To reset the vector store, delete `data/chroma_db/` and `data/ingestion_registry.json`.
- Unit tests can import `utils.config_loader.reload_config()` to point at a test config file.
