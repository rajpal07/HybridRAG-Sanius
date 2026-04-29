import time
import logging
import asyncio
import uuid
from datetime import date
from typing import Optional

import asyncpg
import httpx

from backend.config import GROQ_API_KEY, MAIN_MODEL, NOTES_PROVIDER, NOTES_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY
from backend.database import get_pool, pgvector_available
from backend.privacy.redactor import redact
from backend.ingestion.embedder import embed_text

logger = logging.getLogger(__name__)
BATCH_SIZE = 20

NOTE_TYPES = ["progress note"] * 60 + ["discharge summary"] * 20 + ["nursing note"] * 15 + ["referral letter"] * 5


def _ollama_headers() -> dict:
    if OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return {}


def _generate_note_with_ollama(system_prompt: str, user_prompt: str) -> str:
    # Prefer chat API; fall back to generate API if unavailable.
    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": NOTES_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.post(chat_url, json=payload, headers=_ollama_headers(), timeout=240.0)
            if r.status_code == 404:
                gen_url = f"{OLLAMA_BASE_URL}/api/generate"
                gen_payload = {
                    "model": NOTES_MODEL,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {"temperature": 0.7},
                }
                r = httpx.post(gen_url, json=gen_payload, headers=_ollama_headers(), timeout=240.0)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2)
            else:
                raise
    if last_err is not None and "data" not in locals():
        raise last_err
    if "message" in data and isinstance(data["message"], dict):
        return (data["message"].get("content") or "").strip()
    return (data.get("response") or "").strip()


async def _generate_note_with_ollama_async(system_prompt: str, user_prompt: str) -> str:
    # Run the blocking HTTP work off the event loop so the API stays responsive.
    return await asyncio.to_thread(_generate_note_with_ollama, system_prompt, user_prompt)


async def _redact_async(text: str) -> dict:
    return await asyncio.to_thread(redact, text)


async def _embed_text_async(text: str) -> list[float]:
    return await asyncio.to_thread(embed_text, text)


def _looks_like_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return "rate" in msg and "limit" in msg or "429" in msg


def _pick_note_type(idx: int) -> str:
    return NOTE_TYPES[idx % len(NOTE_TYPES)]


def _age_from_dob(dob: Optional[date], reference_date: Optional[date] = None) -> int:
    if not dob:
        return 45
    ref = reference_date or date.today()
    return ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))


def _build_encounter_prompt(enc: dict) -> str:
    start = enc.get("start_date")
    ref_date = start.date() if hasattr(start, "date") else start
    age = _age_from_dob(enc.get("birth_date"), ref_date)
    date_str = start.strftime("%d %B %Y") if hasattr(start, "strftime") else str(start or "unknown date")
    gender = enc.get("gender", "unknown")
    reason = enc.get("reason_description") or "routine review"
    conditions = enc.get("conditions") or "none recorded"
    medications = enc.get("medications") or "none recorded"
    observations = enc.get("observations") or "none recorded"
    note_type = enc.get("note_type", "progress note")

    return f"""Write a realistic NHS clinical {note_type} for the following patient encounter.
Write in first person past tense. Use natural clinical abbreviations (e.g. Hb, WBC, OD, PRN, NBM).
Include minor spelling variations as a real clinician would. Vary length between 80-300 words.
Ground every clinical detail in the data below — do not invent conditions or medications not listed.

Patient: {age}-year-old {gender}
Encounter date: {date_str}
Encounter reason: {reason}
Active conditions: {conditions}
Current medications: {medications}
Recent observations: {observations}

Write only the clinical note text, no headers or labels."""


async def _fetch_encounter_data(conn: asyncpg.Connection) -> list[dict]:
    rows = await conn.fetch("""
        SELECT
            e.id AS encounter_id,
            e.patient_id,
            e.start_date,
            e.reason_description,
            p.gender,
            p.birth_date,
            (SELECT STRING_AGG(c.condition_description, ', ')
             FROM conditions c WHERE c.patient_id = e.patient_id LIMIT 10) AS conditions,
            (SELECT STRING_AGG(m.drug_name, ', ')
             FROM medications m WHERE m.patient_id = e.patient_id
             AND (m.stop_date IS NULL OR m.stop_date >= e.start_date::date) LIMIT 10) AS medications,
            (SELECT STRING_AGG(t.x, ', ')
             FROM (
                SELECT (o.description || '=' || o.value || o.units) AS x
                FROM observations o
                WHERE o.patient_id = e.patient_id
                ORDER BY o.obs_date DESC
                LIMIT 5
             ) t) AS observations
        FROM encounters e
        JOIN patients p ON p.id = e.patient_id
        WHERE e.id NOT IN (SELECT encounter_id FROM clinical_notes WHERE encounter_id IS NOT NULL)
        ORDER BY e.id
        LIMIT 5000
    """)
    return [dict(r) for r in rows]


async def _ensure_ingestion_runs_table(conn: asyncpg.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
          id TEXT PRIMARY KEY,
          run_type TEXT NOT NULL,
          status TEXT NOT NULL,
          total INTEGER NOT NULL,
          completed INTEGER NOT NULL,
          started_at TIMESTAMP DEFAULT NOW(),
          updated_at TIMESTAMP DEFAULT NOW(),
          error TEXT
        );
        """
    )


async def _set_run(
    conn: asyncpg.Connection,
    run_id: str,
    *,
    status: str,
    total: int,
    completed: int,
    error: str | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO ingestion_runs (id, run_type, status, total, completed, started_at, updated_at, error)
        VALUES ($1, 'notes', $2, $3, $4, NOW(), NOW(), $5)
        ON CONFLICT (id) DO UPDATE SET
            status = EXCLUDED.status,
            total = EXCLUDED.total,
            completed = EXCLUDED.completed,
            updated_at = NOW(),
            error = EXCLUDED.error;
        """,
        run_id,
        status,
        total,
        completed,
        error,
    )


async def generate_notes(*, run_id: str | None = None, limit: int = 5000) -> str:
    use_groq = NOTES_PROVIDER != "ollama"
    client = None
    if use_groq and GROQ_API_KEY:
        from groq import Groq  # noqa: PLC0415
        client = Groq(api_key=GROQ_API_KEY)
    pool = await get_pool()
    run_id = run_id or str(uuid.uuid4())

    async with pool.acquire() as conn:
        await _ensure_ingestion_runs_table(conn)
        use_pgvector = await pgvector_available(conn)

    async with pool.acquire() as conn:
        # Limit is applied here so reruns continue from where they left off.
        encounters = await conn.fetch(
            """
            SELECT
                e.id AS encounter_id,
                e.patient_id,
                e.start_date,
                e.reason_description,
                p.gender,
                p.birth_date,
                (SELECT STRING_AGG(c.condition_description, ', ')
                 FROM conditions c WHERE c.patient_id = e.patient_id LIMIT 10) AS conditions,
                (SELECT STRING_AGG(m.drug_name, ', ')
                 FROM medications m WHERE m.patient_id = e.patient_id
                 AND (m.stop_date IS NULL OR m.stop_date >= e.start_date::date) LIMIT 10) AS medications,
                (SELECT STRING_AGG(t.x, ', ')
                 FROM (
                    SELECT (o.description || '=' || o.value || o.units) AS x
                    FROM observations o
                    WHERE o.patient_id = e.patient_id
                    ORDER BY o.obs_date DESC
                    LIMIT 5
                 ) t) AS observations
            FROM encounters e
            JOIN patients p ON p.id = e.patient_id
            WHERE e.id NOT IN (SELECT encounter_id FROM clinical_notes WHERE encounter_id IS NOT NULL)
            ORDER BY e.id
            LIMIT $1
            """,
            int(limit),
        )
        encounters = [dict(r) for r in encounters]
        await _set_run(conn, run_id, status="running", total=len(encounters), completed=0)

    if not encounters:
        logger.info("No encounters to generate notes for.")
        async with pool.acquire() as conn:
            await _set_run(conn, run_id, status="completed", total=0, completed=0)
        return run_id

    logger.info("Generating notes for %d encounters …", len(encounters))

    system_prompt = (
        "You are an NHS clinician writing realistic clinical notes. "
        "Write in first person past tense. Use natural clinical abbreviations. "
        "Include minor spelling variations. Vary note length from 80-300 words. "
        "Ground every detail in the provided structured data."
    )

    completed = 0
    for batch_start in range(0, len(encounters), BATCH_SIZE):
        batch = encounters[batch_start : batch_start + BATCH_SIZE]

        for idx, enc in enumerate(batch):
            global_idx = batch_start + idx
            note_type = _pick_note_type(global_idx)
            enc["note_type"] = note_type

            try:
                # Heartbeat so status polling shows the job is alive even if one note takes a while.
                if completed == 0 and global_idx == 0:
                    async with pool.acquire() as conn:
                        await _set_run(conn, run_id, status="running", total=len(encounters), completed=completed)

                user_prompt = _build_encounter_prompt(enc)
                note_text = ""
                if client is not None:
                    try:
                        response = await asyncio.to_thread(
                            client.chat.completions.create,
                            model=MAIN_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=600,
                            temperature=0.7,
                        )
                        note_text = response.choices[0].message.content.strip()
                    except Exception as e:
                        if _looks_like_rate_limit(e):
                            logger.warning("Groq rate-limited; falling back to Ollama for note generation")
                            note_text = await _generate_note_with_ollama_async(system_prompt, user_prompt)
                        else:
                            raise
                else:
                    note_text = await _generate_note_with_ollama_async(system_prompt, user_prompt)

                redaction = await _redact_async(note_text)
                redacted = redaction["redacted"]

                embedding = await _embed_text_async(redacted)

                async with pool.acquire() as conn:
                    if use_pgvector:
                        await conn.execute(
                            """INSERT INTO clinical_notes
                                   (patient_id, encounter_id, note_type, content,
                                    redacted_content, embedding)
                               VALUES ($1,$2,$3,$4,$5,$6::vector)""",
                            enc["patient_id"],
                            enc["encounter_id"],
                            note_type,
                            note_text,
                            redacted,
                            str(embedding),
                        )
                    else:
                        await conn.execute(
                            """INSERT INTO clinical_notes
                                   (patient_id, encounter_id, note_type, content,
                                    redacted_content, embedding)
                               VALUES ($1,$2,$3,$4,$5,$6)""",
                            enc["patient_id"],
                            enc["encounter_id"],
                            note_type,
                            note_text,
                            redacted,
                            embedding,
                        )

                completed += 1
                if completed < 25 or completed % 10 == 0 or completed == len(encounters):
                    async with pool.acquire() as conn:
                        await _set_run(conn, run_id, status="running", total=len(encounters), completed=completed)

                if global_idx % 50 == 0:
                    logger.info("  Generated %d / %d notes", global_idx + 1, len(encounters))

            except Exception as e:
                logger.error("Failed to generate note for encounter %s: %s", enc["encounter_id"], e)
                async with pool.acquire() as conn:
                    await _set_run(
                        conn,
                        run_id,
                        status="running",
                        total=len(encounters),
                        completed=completed,
                        error=str(e),
                    )

        # Rate-limit: sleep between batches
        if batch_start + BATCH_SIZE < len(encounters):
            await asyncio.sleep(1)

    async with pool.acquire() as conn:
        await _set_run(conn, run_id, status="completed", total=len(encounters), completed=completed)

    logger.info("Note generation complete.")
    return run_id
