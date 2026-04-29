import os
import logging
from datetime import date
from typing import Optional

import pandas as pd
import asyncpg

from backend.config import RAW_DIR
from backend.database import get_pool

logger = logging.getLogger(__name__)

SYNTHEA_DIR = os.path.join(RAW_DIR, "synthea")
HES_DIR = os.path.join(RAW_DIR, "hes")
PROMS_DIR = os.path.join(RAW_DIR, "proms")

KEY_OBSERVATIONS = {
    "hbf", "hbs", "hemoglobin", "hb", "wbc", "crp", "egfr",
    "pain", "white blood", "c-reactive", "glomerular",
    "glucose", "systolic", "diastolic", "blood pressure",
    "cholesterol", "ldl", "hdl", "bmi", "body mass", "creatinine",
    "a1c", "hba1c", "triglyceride",
}

BATCH_SIZE = 100


def _safe_date(val) -> Optional[date]:
    if pd.isna(val) or val is None:
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _safe_str(val) -> Optional[str]:
    if pd.isna(val) or val is None:
        return None
    return str(val).strip() or None


async def _batch_insert(conn: asyncpg.Connection, query: str, rows: list):
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        await conn.executemany(query, chunk)
        logger.info("  inserted %d / %d rows", min(i + BATCH_SIZE, len(rows)), len(rows))


# ── Synthea loaders ──────────────────────────────────────────────────────────

async def load_patients(conn: asyncpg.Connection):
    path = os.path.join(SYNTHEA_DIR, "patients.csv")
    if not os.path.exists(path):
        logger.warning("patients.csv not found at %s", path)
        return
    df = pd.read_csv(path)
    logger.info("Loading %d patients …", len(df))

    rows = []
    for _, r in df.iterrows():
        rows.append((
            _safe_str(r.get("Id")),
            _safe_str(r.get("GENDER")),
            _safe_date(r.get("BIRTHDATE")),
            _safe_str(r.get("RACE")),
            _safe_str(r.get("ETHNICITY")),
            _safe_float(r.get("LAT")),
            _safe_float(r.get("LON")),
            _safe_str(r.get("CITY")),
            _safe_str(r.get("ORGANIZATION", "Unknown NHS Trust")),
        ))

    await _batch_insert(
        conn,
        """INSERT INTO patients (synthea_id, gender, birth_date, race, ethnicity,
                                  lat, lon, city, trust_name)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
           ON CONFLICT (synthea_id) DO NOTHING""",
        rows,
    )
    logger.info("Patients loaded.")


async def load_encounters(conn: asyncpg.Connection):
    path = os.path.join(SYNTHEA_DIR, "encounters.csv")
    if not os.path.exists(path):
        logger.warning("encounters.csv not found")
        return
    df = pd.read_csv(path)
    logger.info("Loading %d encounters …", len(df))

    # Build synthea_id → db id map
    id_map = dict(await conn.fetch("SELECT synthea_id, id FROM patients"))

    rows = []
    for _, r in df.iterrows():
        pid = id_map.get(_safe_str(r.get("PATIENT")))
        if pid is None:
            continue
        rows.append((
            pid,
            _safe_date(r.get("START")),
            _safe_date(r.get("STOP")),
            _safe_str(r.get("ENCOUNTERCLASS")),
            _safe_str(r.get("REASONDESCRIPTION")),
            _safe_str(r.get("PROVIDER")),
        ))

    await _batch_insert(
        conn,
        """INSERT INTO encounters (patient_id, start_date, end_date,
                                    encounter_class, reason_description, provider)
           VALUES ($1,$2,$3,$4,$5,$6)""",
        rows,
    )
    logger.info("Encounters loaded.")


async def load_conditions(conn: asyncpg.Connection):
    path = os.path.join(SYNTHEA_DIR, "conditions.csv")
    if not os.path.exists(path):
        logger.warning("conditions.csv not found")
        return
    df = pd.read_csv(path)
    logger.info("Loading %d conditions …", len(df))

    id_map = dict(await conn.fetch("SELECT synthea_id, id FROM patients"))

    rows = []
    for _, r in df.iterrows():
        pid = id_map.get(_safe_str(r.get("PATIENT")))
        if pid is None:
            continue
        rows.append((
            pid,
            None,
            _safe_date(r.get("START")),
            _safe_str(r.get("DESCRIPTION")),
            _safe_str(r.get("CODE")),
        ))

    await _batch_insert(
        conn,
        """INSERT INTO conditions (patient_id, encounter_id, onset_date,
                                    condition_description, icd10_code)
           VALUES ($1,$2,$3,$4,$5)""",
        rows,
    )
    logger.info("Conditions loaded.")


async def load_medications(conn: asyncpg.Connection):
    path = os.path.join(SYNTHEA_DIR, "medications.csv")
    if not os.path.exists(path):
        logger.warning("medications.csv not found")
        return
    df = pd.read_csv(path)
    logger.info("Loading %d medications …", len(df))

    id_map = dict(await conn.fetch("SELECT synthea_id, id FROM patients"))

    rows = []
    for _, r in df.iterrows():
        pid = id_map.get(_safe_str(r.get("PATIENT")))
        if pid is None:
            continue
        rows.append((
            pid,
            _safe_date(r.get("START")),
            _safe_date(r.get("STOP")),
            _safe_str(r.get("DESCRIPTION")),
            _safe_str(r.get("REASONDESCRIPTION")),
        ))

    await _batch_insert(
        conn,
        """INSERT INTO medications (patient_id, start_date, stop_date,
                                     drug_name, reason)
           VALUES ($1,$2,$3,$4,$5)""",
        rows,
    )
    logger.info("Medications loaded.")


async def load_observations(conn: asyncpg.Connection):
    path = os.path.join(SYNTHEA_DIR, "observations.csv")
    if not os.path.exists(path):
        logger.warning("observations.csv not found")
        return
    df = pd.read_csv(path)

    # Filter to key lab types
    mask = df["DESCRIPTION"].str.lower().apply(
        lambda d: any(k in str(d).lower() for k in KEY_OBSERVATIONS)
    )
    df = df[mask]
    logger.info("Loading %d filtered observations …", len(df))

    id_map = dict(await conn.fetch("SELECT synthea_id, id FROM patients"))

    rows = []
    for _, r in df.iterrows():
        pid = id_map.get(_safe_str(r.get("PATIENT")))
        if pid is None:
            continue
        rows.append((
            pid,
            None,
            _safe_date(r.get("DATE")),
            _safe_str(r.get("DESCRIPTION")),
            _safe_str(r.get("VALUE")),
            _safe_str(r.get("UNITS")),
        ))

    await _batch_insert(
        conn,
        """INSERT INTO observations (patient_id, encounter_id, obs_date,
                                      description, value, units)
           VALUES ($1,$2,$3,$4,$5,$6)""",
        rows,
    )
    logger.info("Observations loaded.")


# ── NHS HES loader ───────────────────────────────────────────────────────────

async def load_hes(conn: asyncpg.Connection):
    if not os.path.exists(HES_DIR):
        logger.warning("HES directory not found at %s", HES_DIR)
        return

    files = [f for f in os.listdir(HES_DIR) if f.endswith((".xlsx", ".xls", ".csv"))]
    if not files:
        logger.warning("No HES files found in %s", HES_DIR)
        return

    rows = []
    for fname in files:
        fpath = os.path.join(HES_DIR, fname)
        try:
            df = pd.read_excel(fpath) if fname.endswith((".xlsx", ".xls")) else pd.read_csv(fpath)
            # Try to find trust, metric, value, period columns by common names
            col_map = {c.lower(): c for c in df.columns}
            trust_col = next((col_map[k] for k in col_map if "trust" in k or "provider" in k), None)
            value_col = next((col_map[k] for k in col_map if "count" in k or "value" in k or "total" in k), None)
            period_col = next((col_map[k] for k in col_map if "period" in k or "year" in k or "date" in k), None)

            for _, r in df.iterrows():
                trust = _safe_str(r[trust_col]) if trust_col else fname
                val = _safe_float(r[value_col]) if value_col else None
                period_raw = _safe_str(r[period_col]) if period_col else None
                try:
                    period_start = pd.to_datetime(period_raw).date() if period_raw else None
                except Exception:
                    period_start = None

                rows.append((
                    trust or fname,
                    "admission_count",
                    val,
                    period_start,
                    None,
                ))
        except Exception as e:
            logger.error("Failed to load HES file %s: %s", fname, e)

    if rows:
        await _batch_insert(
            conn,
            """INSERT INTO operational_metrics (trust_name, metric_name, metric_value,
                                                 period_start, period_end)
               VALUES ($1,$2,$3,$4,$5)""",
            rows,
        )
        logger.info("HES operational metrics loaded: %d rows", len(rows))


# ── NHS PROMs loader ─────────────────────────────────────────────────────────

async def load_proms(conn: asyncpg.Connection):
    if not os.path.exists(PROMS_DIR):
        logger.warning("PROMs directory not found at %s", PROMS_DIR)
        return

    files = [f for f in os.listdir(PROMS_DIR) if f.endswith(".csv")]
    if not files:
        logger.warning("No PROMs CSV files found in %s", PROMS_DIR)
        return

    # Use patient id=1 as a placeholder when no patient link available
    default_pid_row = await conn.fetchrow("SELECT id FROM patients LIMIT 1")
    default_pid = default_pid_row["id"] if default_pid_row else None

    rows = []
    for fname in files:
        fpath = os.path.join(PROMS_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            col_map = {c.lower(): c for c in df.columns}

            for _, r in df.iterrows():
                def g(keys):
                    for k in keys:
                        for ck in col_map:
                            if k in ck:
                                return _safe_float(r[col_map[ck]])
                    return None

                rows.append((
                    default_pid,
                    "EQ-5D",
                    g(["mobility"]),
                    g(["self_care", "selfcare", "self care"]),
                    g(["usual", "activities"]),
                    g(["pain", "discomfort"]),
                    g(["anxiety", "depression"]),
                    g(["index", "eq5d", "eq_5d"]),
                    _safe_date(r.get(col_map.get("date", col_map.get("period", "")))),
                ))
        except Exception as e:
            logger.error("Failed to load PROMs file %s: %s", fname, e)

    if rows:
        await _batch_insert(
            conn,
            """INSERT INTO pro_responses (patient_id, instrument, score_mobility,
                                          score_self_care, score_activities, score_pain,
                                          score_anxiety, eq5d_index, recorded_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
            rows,
        )
        logger.info("PROMs loaded: %d rows", len(rows))


# ── Main entry point ─────────────────────────────────────────────────────────

async def load_all_csvs():
    pool = await get_pool()
    async with pool.acquire() as conn:
        logger.info("=== Loading Synthea CSVs ===")
        await load_patients(conn)
        await load_encounters(conn)
        await load_conditions(conn)
        await load_medications(conn)
        await load_observations(conn)

        logger.info("=== Loading NHS HES data ===")
        await load_hes(conn)

        logger.info("=== Loading NHS PROMs data ===")
        await load_proms(conn)

    from backend.ingestion.mimic_loader import ingest_mimic

    logger.info("=== Loading MIMIC-III demo data ===")
    await ingest_mimic()

    logger.info("All CSVs loaded successfully.")
