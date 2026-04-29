import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.agents.analytics_agent import analyze
from backend.agents.rag_agent import retrieve
from backend.agents.router import classify_query
from backend.agents.sql_agent import generate_and_run_sql
from backend.agents.synthesizer import synthesize
from backend.config import OLLAMA_BASE_URL
from backend.database import close_pool, get_pool, get_table_counts, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sanius Health Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure all unhandled exceptions return valid JSON so the frontend never hits a JSONDecodeError."""
    logger.error("Unhandled exception on %s: %s\n%s", request.url.path, exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# In-memory async job store for long-running queries
# ---------------------------------------------------------------------------

@dataclass
class _QueryJob:
    job_id: str
    status: str = "running"   # running | done | error
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_JOB_STORE: dict[str, _QueryJob] = {}
_JOB_TTL_SECONDS = 600  # jobs older than 10 min are cleaned up


def _prune_jobs() -> None:
    now = datetime.now(timezone.utc)
    stale = [
        jid for jid, job in _JOB_STORE.items()
        if (now - job.created_at).total_seconds() > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _JOB_STORE[jid]


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Sanius API ready.")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class SessionCreateRequest(BaseModel):
    title: str | None = None


STRONG_FOLLOW_UP_HINTS = (
    "previous",
    "earlier",
    "above",
    "in this chat",
    "from this chat",
    "from the last answer",
    "from the earlier answer",
    "follow up",
    "follow-up",
    "continue",
    "what did you mean",
    "what were we counting",
    "what combination were we counting",
    "you said",
    "last question",
    "last answer",
)

WEAK_FOLLOW_UP_HINTS = (
    "what about",
    "those patients",
    "these patients",
    "that cohort",
    "this cohort",
    "that group",
    "same patient",
    "same cohort",
    "we discussed",
)


def _agents_run(query_type: str) -> list[str]:
    mapping = {
        "sql": ["router", "sql_agent", "synthesizer"],
        "rag": ["router", "rag_agent", "synthesizer"],
        "analytics": ["router", "analytics_agent", "synthesizer"],
        "hybrid": ["router", "sql_agent", "rag_agent", "synthesizer"],
    }
    return mapping.get(query_type, ["router", "synthesizer"])


def _make_session_title(question: str) -> str:
    clean = " ".join((question or "").strip().split())
    if not clean:
        return "New chat"
    return clean[:60] + ("…" if len(clean) > 60 else "")


def _build_session_summary(rows: list) -> str:
    snippets = []
    for row in rows[:8]:
        content = " ".join((row["content"] or "").split())[:120]
        if content:
            snippets.append(f"{row['role']}: {content}")
    return " | ".join(snippets)


async def _refresh_session_summary(conn, session_id: str):
    rows = await conn.fetch(
        """
        WITH ranked AS (
            SELECT role, content, created_at, id,
                   ROW_NUMBER() OVER (ORDER BY created_at DESC, id DESC) AS rn
            FROM chat_messages
            WHERE session_id = $1
        )
        SELECT role, content
        FROM ranked
        WHERE rn > 8
        ORDER BY rn ASC
        LIMIT 8
        """,
        session_id,
    )
    await conn.execute(
        "UPDATE chat_sessions SET summary = $2, updated_at = NOW() WHERE id = $1",
        session_id,
        _build_session_summary(rows),
    )


async def _load_session_context(conn, session_id: str) -> dict | None:
    session = await conn.fetchrow(
        "SELECT id, title, summary, created_at, updated_at FROM chat_sessions WHERE id = $1",
        session_id,
    )
    if not session:
        return None

    message_rows = await conn.fetch(
        """
        SELECT role, content, created_at
        FROM chat_messages
        WHERE session_id = $1
        ORDER BY created_at DESC, id DESC
        LIMIT 8
        """,
        session_id,
    )
    messages = [dict(row) for row in reversed(message_rows)]
    return {"session": dict(session), "messages": messages}


def _contextualize_question(question: str, session_context: dict | None) -> str:
    if not session_context:
        return question

    parts = []
    summary = (session_context.get("session", {}) or {}).get("summary", "").strip()
    if summary:
        parts.append(f"Conversation summary:\n{summary}")

    messages = session_context.get("messages") or []
    if messages:
        lines = []
        for message in messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content']}")
        parts.append("Recent conversation:\n" + "\n".join(lines))

    parts.append(f"Current user question:\n{question}")
    return "\n\n".join(parts)


def _question_needs_session_context(question: str) -> tuple[bool, str]:
    lower = " ".join((question or "").strip().lower().split())
    if not lower:
        return False, "empty_question"
    for hint in STRONG_FOLLOW_UP_HINTS:
        if hint in lower:
            return True, f"matched_hint:{hint}"
    word_count = len(lower.split())
    for hint in WEAK_FOLLOW_UP_HINTS:
        if hint in lower and word_count <= 18:
            return True, f"matched_hint:{hint}"
    tokens = set(lower.split())
    if len(tokens) <= 10 and tokens.intersection({"it", "they", "them", "that", "those", "this", "these"}):
        return True, "short_referential_question"
    return False, "self_contained_question"


def _decode_answer_payload(value):
    if value is None or isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _json_safe(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


async def _save_session_turn(conn, session_id: str, question: str, answer_payload: dict):
    session = await conn.fetchrow("SELECT id, title FROM chat_sessions WHERE id = $1", session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if (session["title"] or "").strip().lower() in {"", "new chat"}:
        await conn.execute(
            "UPDATE chat_sessions SET title = $2, updated_at = NOW() WHERE id = $1",
            session_id,
            _make_session_title(question),
        )

    await conn.execute(
        "INSERT INTO chat_messages (session_id, role, content, answer_payload) VALUES ($1, 'user', $2, NULL)",
        session_id,
        question,
    )
    await conn.execute(
        "INSERT INTO chat_messages (session_id, role, content, answer_payload) VALUES ($1, 'assistant', $2, $3::jsonb)",
        session_id,
        answer_payload.get("answer", ""),
        json.dumps(_json_safe(answer_payload)),
    )
    await _refresh_session_summary(conn, session_id)


async def _list_sessions(conn) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT
            s.id,
            s.title,
            s.summary,
            s.created_at,
            s.updated_at,
            COUNT(m.id) AS message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.id
        GROUP BY s.id
        ORDER BY s.updated_at DESC, s.created_at DESC
        """
    )
    return [dict(row) for row in rows]


def _build_execution_flow(query_type: str, analytics_result: dict | None = None) -> list[str]:
    if query_type == "analytics":
        mode = ((analytics_result or {}).get("plan") or {}).get("execution_mode", "simple")
        flow = ["router", "analytics_planner", "sql_executor"]
        if mode == "multi_step":
            flow.extend(["analytics_reviewer", "analytics_findings"])
        flow.append("synthesizer")
        return flow
    if query_type == "hybrid":
        return ["router", "sql_agent", "rag_agent", "synthesizer"]
    if query_type == "rag":
        return ["router", "rag_agent", "synthesizer"]
    return ["router", "sql_agent", "synthesizer"]


async def _run_query_job(job: _QueryJob, req: QueryRequest) -> None:
    """Background coroutine that executes the full query pipeline and stores the result in the job store."""
    question = req.question.strip()
    try:
        pool = await get_pool()
        session_context = None
        effective_question = question
        session_context_used = False
        session_context_reason = "no_session"

        if req.session_id:
            async with pool.acquire() as conn:
                session_context = await _load_session_context(conn, req.session_id)
                if not session_context:
                    job.status = "error"
                    job.error = "Session not found"
                    return
            session_context_used, session_context_reason = _question_needs_session_context(question)
            if session_context_used:
                effective_question = _contextualize_question(question, session_context)

        router_result = await classify_query(effective_question)
        query_type = router_result.get("type", "hybrid")
        logger.info("[job %s] Query classified as: %s", job.job_id[:8], query_type)

        sql_result = None
        rag_result = None
        analytics_result = None

        async with pool.acquire() as conn:
            if query_type == "sql":
                sql_result = await generate_and_run_sql(effective_question, conn)
            elif query_type == "rag":
                rag_result = await retrieve(effective_question, conn)
            elif query_type == "analytics":
                try:
                    analytics_result = await asyncio.wait_for(
                        analyze(effective_question, conn),
                        timeout=300.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[job %s] Analytics timed out after 300s", job.job_id[:8])
                    analytics_result = {
                        "error": "Analytics timed out after 300s. The LLM may be under load — please retry.",
                        "partial": True,
                        "key_findings": [],
                        "limitations": ["Query exceeded 300s time limit."],
                        "plan": {}, "review": {}, "step_results": [],
                        "summary_stats": {}, "data_table": [], "chart_data": {},
                        "missing_requirements": [],
                    }
            elif query_type == "hybrid":
                try:
                    sql_result = await generate_and_run_sql(effective_question, conn)
                except Exception as exc:  # noqa: BLE001
                    logger.error("[job %s] SQL agent error: %s", job.job_id[:8], exc)
                    sql_result = None
                try:
                    rag_result = await retrieve(effective_question, conn)
                except Exception as exc:  # noqa: BLE001
                    logger.error("[job %s] RAG agent error: %s", job.job_id[:8], exc)
                    rag_result = None

        final = await synthesize(
            query=question,
            router_result=router_result,
            sql_result=sql_result,
            rag_result=rag_result,
            analytics_result=analytics_result,
        )

        response_payload = {
            "question": question,
            "session_id": req.session_id,
            "effective_question": effective_question,
            "session_context_used": session_context_used,
            "session_context_reason": session_context_reason,
            "router": router_result,
            "answer": final["answer"],
            "citations": final["citations"],
            "confidence": final["confidence"],
            "sources_used": final["sources_used"],
            "sql_used": final.get("sql_used"),
            "sql_results": sql_result.get("results", []) if sql_result else [],
            "sql_row_count": sql_result.get("row_count", 0) if sql_result else 0,
            "rag_chunks": rag_result.get("chunks", []) if rag_result else [],
            "analytics": analytics_result,
            "agents_run": _agents_run(query_type),
            "execution_flow": _build_execution_flow(query_type, analytics_result),
        }

        if req.session_id:
            async with pool.acquire() as conn:
                await _save_session_turn(conn, req.session_id, question, response_payload)

        job.result = response_payload
        job.status = "done"
        logger.info("[job %s] Completed successfully", job.job_id[:8])

    except Exception as exc:  # noqa: BLE001
        logger.error("[job %s] Failed: %s\n%s", job.job_id[:8], exc, traceback.format_exc())
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"


@app.post("/query")
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    """Accept a query and return a job_id immediately. The frontend polls /query/status/{job_id}."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    _prune_jobs()
    job_id = str(uuid.uuid4())
    job = _QueryJob(job_id=job_id)
    _JOB_STORE[job_id] = job

    background_tasks.add_task(_run_query_job, job, req)
    return {"job_id": job_id, "status": "running"}


@app.get("/query/status/{job_id}")
async def query_status(job_id: str):
    """Poll for the result of a background query job."""
    job = _JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job.status == "running":
        return {"job_id": job_id, "status": "running"}
    if job.status == "error":
        return {"job_id": job_id, "status": "error", "error": job.error}
    return {"job_id": job_id, "status": "done", "result": job.result}


@app.get("/sessions")
async def list_sessions():
    pool = await get_pool()
    async with pool.acquire() as conn:
        sessions = await _list_sessions(conn)
    return {"status": "ok", "sessions": sessions}


@app.post("/sessions")
async def create_session(req: SessionCreateRequest | None = None):
    session_id = str(uuid.uuid4())
    title = (req.title.strip() if req and req.title else "") or "New chat"
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_sessions (id, title) VALUES ($1, $2)",
            session_id,
            title,
        )
        session = await conn.fetchrow(
            "SELECT id, title, summary, created_at, updated_at FROM chat_sessions WHERE id = $1",
            session_id,
        )
    return {"status": "ok", "session": dict(session)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        session = await conn.fetchrow(
            "SELECT id, title, summary, created_at, updated_at FROM chat_sessions WHERE id = $1",
            session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        message_rows = await conn.fetch(
            """
            SELECT role, content, answer_payload, created_at
            FROM chat_messages
            WHERE session_id = $1
            ORDER BY created_at ASC, id ASC
            """,
            session_id,
        )

    messages = []
    for row in message_rows:
        item = dict(row)
        item["answer_payload"] = _decode_answer_payload(item.get("answer_payload"))
        messages.append(item)

    return {"status": "ok", "session": dict(session), "messages": messages}


@app.post("/ingest/csv")
async def ingest_csv():
    from backend.ingestion.csv_loader import load_all_csvs

    try:
        await load_all_csvs()
        return {"status": "ok", "message": "CSV ingestion complete (Synthea, HES, PROMs, MIMIC)"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/mimic")
async def ingest_mimic():
    from backend.ingestion.mimic_loader import ingest_mimic as _ingest_mimic

    try:
        result = await _ingest_mimic()
        return {"status": "ok", "message": "MIMIC ingestion complete", "result": result}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/pdfs")
async def ingest_pdfs():
    from backend.ingestion.pdf_parser import ingest_all_pdfs

    try:
        await ingest_all_pdfs()
        return {"status": "ok", "message": "PDF ingestion complete"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/ehg")
async def ingest_ehg():
    from backend.ingestion.ehg_loader import ingest_ehg as _ingest_ehg

    try:
        result = await _ingest_ehg()
        return {"status": "ok", "message": "EHG ingestion complete", "result": result}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/notes")
async def ingest_notes(
    background_tasks: BackgroundTasks,
    limit: int = Query(5000, ge=1, le=5000),
    background: bool = Query(True),
    run_id: str | None = Query(None),
):
    from backend.ingestion.note_generator import generate_notes

    try:
        rid = run_id or str(uuid.uuid4())
        if background:
            background_tasks.add_task(generate_notes, run_id=rid, limit=limit)
            return {"status": "ok", "message": "Note generation started", "run_id": rid, "limit": limit}

        rid = await generate_notes(run_id=rid, limit=limit)
        return {"status": "ok", "message": "Note generation complete", "run_id": rid, "limit": limit}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ingest/notes/status/{run_id}")
async def ingest_notes_status(run_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, status, total, completed, started_at, updated_at, error
            FROM ingestion_runs
            WHERE id = $1 AND run_type = 'notes'
            """,
            run_id,
        )
        notes_total = await conn.fetchval("SELECT COUNT(*) FROM clinical_notes")
        remaining = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM encounters e
            WHERE e.id NOT IN (SELECT encounter_id FROM clinical_notes WHERE encounter_id IS NOT NULL)
            """
        )

    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    run_dict = dict(row)
    if run_dict.get("status") == "running":
        try:
            updated_at = run_dict.get("updated_at")
            if isinstance(updated_at, datetime):
                if updated_at.tzinfo is None:
                    age_sec = (datetime.now() - updated_at).total_seconds()
                else:
                    age_sec = (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds()
                run_dict["seconds_since_update"] = int(age_sec)
                if age_sec > 600:
                    run_dict["stale"] = True
        except Exception:
            pass
        async with pool.acquire() as conn:
            est = await conn.fetchval(
                "SELECT COUNT(*) FROM clinical_notes WHERE created_at >= $1",
                run_dict.get("started_at"),
            )
        run_dict["estimated_completed"] = max(int(run_dict.get("completed") or 0), int(est or 0))

    return {
        "status": "ok",
        "run": run_dict,
        "clinical_notes_total": notes_total,
        "encounters_remaining_for_notes": remaining,
    }


@app.get("/stats")
async def stats():
    try:
        return {"status": "ok", "counts": await get_table_counts()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    db_status = "connected"
    ollama_status = "connected"

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                ollama_status = f"error: HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001
        ollama_status = f"error: {exc}"

    from backend.config import GROQ_API_KEY, LLAMA_PARSE_API_KEY  # noqa: PLC0415

    groq_status = "not configured"
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                )
                groq_status = "connected" if r.status_code == 200 else f"error: HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            groq_status = f"error: {exc}"

    llamaparse_status = "not configured"
    if LLAMA_PARSE_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://api.cloud.llamaindex.ai/api/parsing/job",
                    headers={"Authorization": f"Bearer {LLAMA_PARSE_API_KEY}"},
                )
                llamaparse_status = "connected" if r.status_code in (200, 400, 404) else f"error: HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            llamaparse_status = f"error: {exc}"

    return {"status": "ok", "db": db_status, "ollama": ollama_status, "groq": groq_status, "llamaparse": llamaparse_status}
