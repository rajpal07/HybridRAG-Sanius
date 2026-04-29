import asyncpg
import logging
from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool = None
_pgvector_available: bool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def pgvector_available(conn: asyncpg.Connection | None = None) -> bool:
    global _pgvector_available
    if _pgvector_available is not None:
        return _pgvector_available

    try:
        if conn is None:
            pool = await get_pool()
            async with pool.acquire() as c:
                row = await c.fetchrow(
                    "select 1 as ok from pg_available_extensions where name='vector' limit 1"
                )
        else:
            row = await conn.fetchrow(
                "select 1 as ok from pg_available_extensions where name='vector' limit 1"
            )
        _pgvector_available = bool(row)
    except Exception as e:
        logger.warning("pgvector availability check failed (%s); assuming unavailable", e)
        _pgvector_available = False

    return _pgvector_available


CREATE_TABLES_PGVECTOR_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    synthea_id TEXT UNIQUE,
    gender TEXT,
    birth_date DATE,
    race TEXT,
    ethnicity TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    city TEXT,
    trust_name TEXT
);

CREATE TABLE IF NOT EXISTS encounters (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    encounter_class TEXT,
    reason_description TEXT,
    provider TEXT
);

CREATE TABLE IF NOT EXISTS conditions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    encounter_id INTEGER REFERENCES encounters(id),
    onset_date DATE,
    condition_description TEXT,
    icd10_code TEXT
);

CREATE TABLE IF NOT EXISTS medications (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    start_date DATE,
    stop_date DATE,
    drug_name TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    encounter_id INTEGER REFERENCES encounters(id),
    obs_date TIMESTAMP,
    description TEXT,
    value TEXT,
    units TEXT
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    encounter_id INTEGER REFERENCES encounters(id),
    note_type TEXT,
    content TEXT,
    redacted_content TEXT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source_name TEXT,
    doc_type TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding vector(768),
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pro_responses (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    instrument TEXT,
    score_mobility DOUBLE PRECISION,
    score_self_care DOUBLE PRECISION,
    score_activities DOUBLE PRECISION,
    score_pain DOUBLE PRECISION,
    score_anxiety DOUBLE PRECISION,
    eq5d_index DOUBLE PRECISION,
    recorded_at DATE
);

CREATE TABLE IF NOT EXISTS operational_metrics (
    id SERIAL PRIMARY KEY,
    trust_name TEXT,
    metric_name TEXT,
    metric_value DOUBLE PRECISION,
    period_start DATE,
    period_end DATE
);

CREATE TABLE IF NOT EXISTS ehg_records (
    id SERIAL PRIMARY KEY,
    record_name TEXT UNIQUE,
    rec_id INTEGER,
    gestation_weeks DOUBLE PRECISION,
    preterm BOOLEAN,
    rectime_minutes DOUBLE PRECISION,
    maternal_age INTEGER,
    parity INTEGER,
    abortions INTEGER,
    weight_kg DOUBLE PRECISION,
    hypertension BOOLEAN,
    diabetes BOOLEAN,
    placental_position TEXT,
    bleeding_first_trimester BOOLEAN,
    bleeding_second_trimester BOOLEAN,
    funneling TEXT,
    smoker BOOLEAN,
    n_signals INTEGER,
    sample_rate DOUBLE PRECISION,
    n_samples INTEGER,
    header_path TEXT,
    data_path TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ehg_features (
    id SERIAL PRIMARY KEY,
    record_name TEXT NOT NULL REFERENCES ehg_records(record_name) ON DELETE CASCADE,
    channel INTEGER NOT NULL,
    variant TEXT NOT NULL, -- raw / DOCFILT-4-0.08-4 / DOCFILT-4-0.3-3 / DOCFILT-4-0.3-4
    sample_rate DOUBLE PRECISION,
    n_samples INTEGER,
    mean DOUBLE PRECISION,
    std DOUBLE PRECISION,
    rms DOUBLE PRECISION,
    p2p DOUBLE PRECISION,
    skew DOUBLE PRECISION,
    kurtosis DOUBLE PRECISION,
    bandpower_0_08_0_3 DOUBLE PRECISION,
    bandpower_0_3_1_0 DOUBLE PRECISION,
    bandpower_1_0_3_0 DOUBLE PRECISION,
    bandpower_3_0_4_0 DOUBLE PRECISION,
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(record_name, channel, variant)
);

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

CREATE INDEX IF NOT EXISTS idx_clinical_notes_embedding
    ON clinical_notes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


CREATE_TABLES_NO_PGVECTOR_SQL = """
-- pgvector extension is not installed in this Postgres instance.
-- Use a schema that keeps the app runnable without vector search.

CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    synthea_id TEXT UNIQUE,
    gender TEXT,
    birth_date DATE,
    race TEXT,
    ethnicity TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    city TEXT,
    trust_name TEXT
);

CREATE TABLE IF NOT EXISTS encounters (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    encounter_class TEXT,
    reason_description TEXT,
    provider TEXT
);

CREATE TABLE IF NOT EXISTS conditions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    encounter_id INTEGER REFERENCES encounters(id),
    onset_date DATE,
    condition_description TEXT,
    icd10_code TEXT
);

CREATE TABLE IF NOT EXISTS medications (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    start_date DATE,
    stop_date DATE,
    drug_name TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    encounter_id INTEGER REFERENCES encounters(id),
    obs_date TIMESTAMP,
    description TEXT,
    value TEXT,
    units TEXT
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    encounter_id INTEGER REFERENCES encounters(id),
    note_type TEXT,
    content TEXT,
    redacted_content TEXT,
    embedding DOUBLE PRECISION[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source_name TEXT,
    doc_type TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding DOUBLE PRECISION[],
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pro_responses (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    instrument TEXT,
    score_mobility DOUBLE PRECISION,
    score_self_care DOUBLE PRECISION,
    score_activities DOUBLE PRECISION,
    score_pain DOUBLE PRECISION,
    score_anxiety DOUBLE PRECISION,
    eq5d_index DOUBLE PRECISION,
    recorded_at DATE
);

CREATE TABLE IF NOT EXISTS operational_metrics (
    id SERIAL PRIMARY KEY,
    trust_name TEXT,
    metric_name TEXT,
    metric_value DOUBLE PRECISION,
    period_start DATE,
    period_end DATE
);

CREATE TABLE IF NOT EXISTS ehg_records (
    id SERIAL PRIMARY KEY,
    record_name TEXT UNIQUE,
    rec_id INTEGER,
    gestation_weeks DOUBLE PRECISION,
    preterm BOOLEAN,
    rectime_minutes DOUBLE PRECISION,
    maternal_age INTEGER,
    parity INTEGER,
    abortions INTEGER,
    weight_kg DOUBLE PRECISION,
    hypertension BOOLEAN,
    diabetes BOOLEAN,
    placental_position TEXT,
    bleeding_first_trimester BOOLEAN,
    bleeding_second_trimester BOOLEAN,
    funneling TEXT,
    smoker BOOLEAN,
    n_signals INTEGER,
    sample_rate DOUBLE PRECISION,
    n_samples INTEGER,
    header_path TEXT,
    data_path TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ehg_features (
    id SERIAL PRIMARY KEY,
    record_name TEXT NOT NULL REFERENCES ehg_records(record_name) ON DELETE CASCADE,
    channel INTEGER NOT NULL,
    variant TEXT NOT NULL,
    sample_rate DOUBLE PRECISION,
    n_samples INTEGER,
    mean DOUBLE PRECISION,
    std DOUBLE PRECISION,
    rms DOUBLE PRECISION,
    p2p DOUBLE PRECISION,
    skew DOUBLE PRECISION,
    kurtosis DOUBLE PRECISION,
    bandpower_0_08_0_3 DOUBLE PRECISION,
    bandpower_0_3_1_0 DOUBLE PRECISION,
    bandpower_1_0_3_0 DOUBLE PRECISION,
    bandpower_3_0_4_0 DOUBLE PRECISION,
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(record_name, channel, variant)
);

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

CREATE_MIMIC_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS mimic_patients (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER UNIQUE,
    gender TEXT,
    dob TIMESTAMP,
    dod TIMESTAMP,
    dod_hosp TIMESTAMP,
    dod_ssn TIMESTAMP,
    expire_flag BOOLEAN
);

CREATE TABLE IF NOT EXISTS mimic_admissions (
    id SERIAL PRIMARY KEY,
    hadm_id INTEGER UNIQUE,
    subject_id INTEGER REFERENCES mimic_patients(subject_id),
    admittime TIMESTAMP,
    dischtime TIMESTAMP,
    deathtime TIMESTAMP,
    admission_type TEXT,
    admission_location TEXT,
    discharge_location TEXT,
    insurance TEXT,
    language TEXT,
    religion TEXT,
    marital_status TEXT,
    ethnicity TEXT,
    edregtime TIMESTAMP,
    edouttime TIMESTAMP,
    diagnosis TEXT,
    hospital_expire_flag BOOLEAN,
    has_chartevents_data BOOLEAN
);

CREATE TABLE IF NOT EXISTS mimic_icustays (
    id SERIAL PRIMARY KEY,
    icustay_id INTEGER UNIQUE,
    subject_id INTEGER REFERENCES mimic_patients(subject_id),
    hadm_id INTEGER REFERENCES mimic_admissions(hadm_id),
    dbsource TEXT,
    first_careunit TEXT,
    last_careunit TEXT,
    first_wardid INTEGER,
    last_wardid INTEGER,
    intime TIMESTAMP,
    outtime TIMESTAMP,
    los DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS mimic_d_icd_diagnoses (
    id SERIAL PRIMARY KEY,
    icd9_code TEXT UNIQUE,
    short_title TEXT,
    long_title TEXT
);

CREATE TABLE IF NOT EXISTS mimic_diagnoses_icd (
    id SERIAL PRIMARY KEY,
    row_id INTEGER UNIQUE,
    subject_id INTEGER REFERENCES mimic_patients(subject_id),
    hadm_id INTEGER REFERENCES mimic_admissions(hadm_id),
    seq_num INTEGER,
    icd9_code TEXT
);

CREATE TABLE IF NOT EXISTS mimic_d_labitems (
    id SERIAL PRIMARY KEY,
    itemid INTEGER UNIQUE,
    label TEXT,
    fluid TEXT,
    category TEXT,
    loinc_code TEXT
);

CREATE TABLE IF NOT EXISTS mimic_labevents (
    id SERIAL PRIMARY KEY,
    row_id INTEGER UNIQUE,
    subject_id INTEGER REFERENCES mimic_patients(subject_id),
    hadm_id INTEGER REFERENCES mimic_admissions(hadm_id),
    itemid INTEGER REFERENCES mimic_d_labitems(itemid),
    charttime TIMESTAMP,
    value TEXT,
    valuenum DOUBLE PRECISION,
    valueuom TEXT,
    flag TEXT
);

CREATE TABLE IF NOT EXISTS mimic_prescriptions (
    id SERIAL PRIMARY KEY,
    row_id INTEGER UNIQUE,
    subject_id INTEGER REFERENCES mimic_patients(subject_id),
    hadm_id INTEGER REFERENCES mimic_admissions(hadm_id),
    icustay_id INTEGER,
    startdate TIMESTAMP,
    enddate TIMESTAMP,
    drug_type TEXT,
    drug TEXT,
    drug_name_poe TEXT,
    drug_name_generic TEXT,
    formulary_drug_cd TEXT,
    gsn TEXT,
    ndc TEXT,
    prod_strength TEXT,
    dose_val_rx TEXT,
    dose_unit_rx TEXT,
    form_val_disp TEXT,
    form_unit_disp TEXT,
    route TEXT
);

CREATE TABLE IF NOT EXISTS mimic_admission_features (
    id SERIAL PRIMARY KEY,
    hadm_id INTEGER UNIQUE REFERENCES mimic_admissions(hadm_id),
    subject_id INTEGER REFERENCES mimic_patients(subject_id),
    age_at_admit DOUBLE PRECISION,
    icu_stay_count INTEGER,
    icu_los_days DOUBLE PRECISION,
    has_diabetes BOOLEAN,
    has_hypertension BOOLEAN,
    hospital_expire_flag BOOLEAN,
    diagnosis_text TEXT
);
"""

CREATE_CHAT_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New chat',
    summary TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    answer_payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages (session_id, created_at);
"""


async def init_db():
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            schema = CREATE_TABLES_PGVECTOR_SQL if await pgvector_available(conn) else CREATE_TABLES_NO_PGVECTOR_SQL
            await conn.execute(schema)
            # Idempotent migration: add metadata column to existing databases
            await conn.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'"
            )
            await conn.execute(CREATE_MIMIC_TABLES_SQL)
            await conn.execute(CREATE_CHAT_TABLES_SQL)
            await conn.execute(
                "ALTER TABLE mimic_diagnoses_icd DROP CONSTRAINT IF EXISTS mimic_diagnoses_icd_icd9_code_fkey"
            )
            logger.info("Database tables initialised successfully")
        except Exception as e:
            logger.error(f"Error initialising database: {e}")
            raise


async def get_table_counts() -> dict:
    pool = await get_pool()
    tables = [
        "patients", "encounters", "conditions", "medications",
        "observations", "clinical_notes", "documents",
        "pro_responses", "operational_metrics",
        "ehg_records",
        "ehg_features",
        "mimic_patients",
        "mimic_admissions",
        "mimic_icustays",
        "mimic_diagnoses_icd",
        "mimic_labevents",
        "mimic_prescriptions",
        "mimic_admission_features",
    ]
    counts = {}
    async with pool.acquire() as conn:
        for table in tables:
            try:
                row = await conn.fetchrow(f"SELECT COUNT(*) AS cnt FROM {table}")
                counts[table] = row["cnt"]
            except Exception:
                counts[table] = 0
    return counts
