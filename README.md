# HybridRAG Sanius

A multi-agent retrieval system that answers natural-language clinical questions across structured EHR records, unstructured documents, biomedical signals, and clinical guidelines — in one pass, with citations and a calibrated confidence score.

The system runs on FastAPI + Streamlit, backed by PostgreSQL with pgvector. A router classifies each question, the relevant agents (SQL, RAG, analytics) run in parallel, and a synthesizer fuses every branch into a single grounded answer with a replayable reasoning trace.

---

## What it does

- **Hybrid retrieval in one pass.** One question, three agents, one answer. EHR tables, guideline documents, and computed analytics fused together with citations.
- **Cross-dataset reasoning.** Validated joins between Synthea community-care records and MIMIC-III ICU records — does an outpatient gap surface as an acute outcome?
- **Self-correcting SQL agent.** Retries on errors and on zero-row results with format hints. Capped to predictable cost.
- **Planner / executor / reviewer loop on analytics.** Complex questions are decomposed, executed, judged for sufficiency, and extended with follow-up steps before any answer is drafted.
- **Domain-tagged document retrieval.** PDFs are auto-tagged at ingest with clinical-domain metadata; query-time filtering keeps an off-topic guideline from polluting an answer.
- **Empty-context abstain.** When evidence is missing, the system explains the gap rather than guessing.
- **PHI safety by design.** Clinical notes are redacted via Microsoft Presidio + spaCy before retrieval; the model only ever sees the de-identified copy.
- **Per-session memory.** Each chat keeps its own conversation context for follow-up questions.
- **Reasoning trace surfaced to the user.** Query type, agents that ran, queries executed, document chunks retrieved, plan steps — every answer is auditable in the UI.

## Datasets

| Dataset | What it is |
|---|---|
| **Synthea EHR** | Synthetic but realistic outpatient/community records — patients, encounters, conditions, medications, observations, PROs, operational metrics. |
| **MIMIC-III demo** | Real anonymised ICU data — admissions, ICU stays, lab events, prescriptions, diagnoses (ICD-9), and a pre-joined admission-features summary. |
| **EHG signals** | Electrohysterography time-series and extracted features for preterm-birth risk research. |
| **Clinical guideline PDFs** | NICE / NHS guidance, parsed via LlamaParse and chunked with LLM-extracted JSONB metadata for domain-aware retrieval. |

## Architecture

```
User question
  → POST /query (returns job_id)
  → router            classify: sql | rag | analytics | hybrid
  → agents in parallel
      sql_agent       generates + executes PostgreSQL with self-correcting retries
      rag_agent       embeds query (nomic-embed-text), domain-filters, vector search + LLM rerank, FTS fallback
      analytics_agent planner → executor → reviewer → follow-up loop
  → synthesizer       fuses results with citations, confidence, and abstain-on-empty
  → /query/status/{job_id}  poll until done
```

Two storage paths — pgvector when available, full-text-search fallback when not. Same agent code, no environment lock-in.

Provider routing — small, cheap, or open-weight models for routing/classification/rerank; frontier models for narration. Auto-fallback on rate-limit, no failed queries.

## Stack

- **Backend** — FastAPI, asyncpg, httpx
- **Database** — PostgreSQL 15 + pgvector (with FTS fallback)
- **Models** — Groq (router + main) with Ollama fallback; nomic-embed-text for embeddings
- **PDF parsing** — LlamaParse with PyMuPDF fallback
- **PHI redaction** — Microsoft Presidio + spaCy `en_core_web_lg`
- **Frontend** — Streamlit

## Quickstart

### 1. Postgres

```bash
docker compose up -d
```

The provided `docker-compose.yml` brings up PostgreSQL 15 with pgvector available.

### 2. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Configuration

Copy `.env.example` to `.env` and fill in the values:

```env
DATABASE_URL=postgresql://sanius:sanius@localhost:5432/sanius
GROQ_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=...                 # only required for Ollama Cloud
LLAMA_PARSE_API_KEY=...

LLM_PROVIDER=auto                  # 'auto' picks Groq if key present, else Ollama
LLM_FALLBACK_PROVIDER=ollama
MAIN_MODEL=llama-3.3-70b-versatile
ROUTER_MODEL=llama-3.1-8b-instant
OLLAMA_MAIN_MODEL=qwen3.5:cloud
EMBED_MODEL=nomic-embed-text

MIMIC_SOURCE_DIR=data/raw/mimiciii_demo/unzipped/mimic-iii-clinical-database-demo-1.4
EHG_SOURCE_DIR=data/raw/hes
```

### 4. Source data

Place raw data under `data/raw/`:

```
data/raw/
├── synthea/                # Synthea CSVs
├── mimiciii_demo/          # MIMIC-III demo dump
├── hes/                    # EHG signal recordings
└── pdfs/                   # NICE / NHS guideline PDFs
```

### 5. Run

```bash
# API on :8000
uvicorn backend.api:app --reload

# UI on :8501 (separate terminal)
streamlit run frontend/app.py
```

Then open the Streamlit UI, click the ingest buttons in the sidebar (or hit the endpoints below), and start asking questions.

## Ingestion endpoints

| Endpoint | What it loads |
|---|---|
| `POST /ingest/csv` | Synthea CSVs from `data/raw/synthea/` |
| `POST /ingest/mimic` | MIMIC-III demo tables from `MIMIC_SOURCE_DIR` |
| `POST /ingest/ehg` | EHG records and extracted features from `EHG_SOURCE_DIR` |
| `POST /ingest/pdfs` | PDFs from `data/raw/pdfs/` — LLM extracts domain metadata per chunk |
| `POST /ingest/notes` | Generates synthetic NHS-style clinical notes for every Synthea encounter |

Re-ingesting PDFs requires clearing existing chunks first (`DELETE FROM documents`) — the parser does not upsert.

## Project layout

```
backend/
  api.py                FastAPI app, routes, job store
  llm.py                Provider-agnostic chat() with auto-fallback
  database.py           Schema bootstrap, pgvector / FTS detection
  config.py             Env loading
  agents/
    router.py           Query classifier
    sql_agent.py        NL → SQL with self-correction
    rag_agent.py        Vector retrieval + LLM rerank + FTS fallback
    analytics_agent.py  Planner / executor / reviewer loop
    synthesizer.py      Composer with citations + confidence + abstain
  ingestion/
    csv_loader.py       Synthea CSV → tables
    mimic_loader.py     MIMIC-III demo → tables
    ehg_loader.py       EHG records + features
    pdf_parser.py       LlamaParse / PyMuPDF + domain metadata extraction
    note_generator.py   Synthetic NHS clinical notes via LLM
    embedder.py         nomic-embed-text wrapper
  privacy/
    redactor.py         Presidio + spaCy PHI redaction

frontend/
  app.py                Streamlit chat UI with reasoning trace expanders

scripts/
  doctor.py             Environment check
  download_sources.py   Helper for source-data download
```

## Health check

```bash
python scripts/doctor.py
```

Validates database connectivity, model providers, embedding endpoint, and PHI redactor.


