import logging
import asyncpg
import time
from backend.config import MAIN_MODEL, OLLAMA_MAIN_MODEL
from backend.llm import chat

logger = logging.getLogger(__name__)

SCHEMA_DESCRIPTION = """
Database schema (PostgreSQL) — Synthea synthetic EHR data, date range approximately 2010–2024.

patients(id, synthea_id, gender, birth_date, race, ethnicity, lat, lon, city, trust_name)
  - gender: 'M' or 'F'
  - birth_date: DATE. Age = DATE_PART('year', AGE(birth_date))

encounters(id, patient_id, start_date, end_date, encounter_class, reason_description, provider)
  - encounter_class values: 'ambulatory', 'emergency', 'inpatient', 'wellness', 'outpatient'

conditions(id, patient_id, encounter_id, onset_date, condition_description, icd10_code)
  - CRITICAL: icd10_code stores SNOMED CT numeric codes (e.g. '44054006'), NOT ICD-10 codes.
    NEVER filter with icd10_code under ANY circumstances — it will ALWAYS return 0 rows.
    These patterns ALL return 0 rows and must NEVER be used:
      icd10_code ILIKE '%E11%'   (diabetes)
      icd10_code ILIKE '%N18%'   (CKD)
      icd10_code ILIKE '%N19%'   (renal failure)
      icd10_code ILIKE '%I10%'   (hypertension)
      icd10_code ILIKE '%I25%'   (coronary artery disease)
    ALWAYS filter using: condition_description ILIKE '%keyword%'
  - CONDITION → ILIKE PATTERNS:
      Diabetes type 2      : condition_description ILIKE '%diabetes%'
      Hypertension         : condition_description ILIKE '%hypertension%'
      CKD / renal          : condition_description ILIKE '%chronic kidney disease%' OR ILIKE '%renal insufficiency%' OR ILIKE '%kidney failure%'
      Heart failure        : condition_description ILIKE '%heart failure%'
      Coronary artery dis. : condition_description ILIKE '%coronary heart disease%' OR ILIKE '%ischemic heart%'
      Atrial fibrillation  : condition_description ILIKE '%atrial fibrillation%'
      COPD                 : condition_description ILIKE '%chronic obstructive%' OR ILIKE '%COPD%'
      Asthma               : condition_description ILIKE '%asthma%'
      Obesity              : condition_description ILIKE '%obesity%' OR ILIKE '%obese%'
      Depression           : condition_description ILIKE '%depression%'
      Stroke               : condition_description ILIKE '%stroke%' OR ILIKE '%cerebrovascular%'

medications(id, patient_id, start_date, stop_date, drug_name, reason)
  - stop_date NULL = no recorded stop date (open prescription as of data snapshot)
  - Active at a given encounter: stop_date IS NULL OR stop_date >= encounter.start_date::date
  - NEVER use drug class names in ILIKE (e.g. '%ACE inhibitor%', '%NSAID%', '%statin%' will return 0 rows).
    Synthea stores full drug names like "Lisinopril 10 MG Oral Tablet". Always match on the molecule name.
  - DRUG CLASS → ILIKE PATTERNS (use OR across all members of the class):
      ACE inhibitors : '%lisinopril%' OR '%ramipril%' OR '%enalapril%' OR '%perindopril%' OR '%captopril%' OR '%benazepril%' OR '%quinapril%' OR '%fosinopril%'
                       IMPORTANT: use '%enalapril%' NOT '%benalapril%' — benalapril does not exist as a drug
      ARBs           : '%losartan%' OR '%valsartan%' OR '%irbesartan%' OR '%candesartan%' OR '%olmesartan%' OR '%telmisartan%'
      NSAIDs         : '%ibuprofen%' OR '%naproxen%' OR '%diclofenac%' OR '%meloxicam%' OR '%indomethacin%' OR '%celecoxib%' OR '%ketorolac%'
      Statins        : '%atorvastatin%' OR '%simvastatin%' OR '%rosuvastatin%' OR '%pravastatin%' OR '%fluvastatin%'
      Beta-blockers  : '%metoprolol%' OR '%atenolol%' OR '%bisoprolol%' OR '%carvedilol%' OR '%propranolol%'
      Diuretics      : '%furosemide%' OR '%hydrochlorothiazide%' OR '%spironolactone%' OR '%bumetanide%' OR '%indapamide%'
      Metformin      : '%metformin%'
      Insulin        : '%insulin%'
      Anticoagulants : '%warfarin%' OR '%apixaban%' OR '%rivaroxaban%' OR '%dabigatran%' OR '%heparin%'
      Antiplatelets  : '%aspirin%' OR '%clopidogrel%' OR '%ticagrelor%'
      Opioids        : '%oxycodone%' OR '%morphine%' OR '%codeine%' OR '%tramadol%' OR '%fentanyl%' OR '%hydrocodone%'
  - Concurrent prescriptions (overlapping periods):
      m1.start_date <= COALESCE(m2.stop_date, m1.start_date + INTERVAL '1 day')
      AND m2.start_date <= COALESCE(m1.stop_date, m2.start_date + INTERVAL '1 day')
  - 'Not on a drug' exclusion MUST use NOT EXISTS with proper parentheses. Example: "Not on Metformin or Insulin"
      NOT EXISTS (
        SELECT 1 FROM medications m_sub 
        WHERE m_sub.patient_id = patients.id 
        AND (m_sub.drug_name ILIKE '%metformin%' OR m_sub.drug_name ILIKE '%insulin%')
      )
  - DATA IS HISTORICAL. NEVER use CURRENT_DATE. Anchor recent windows to:
    (SELECT MAX(start_date) FROM medications) - INTERVAL '12 months'

observations(id, patient_id, encounter_id, obs_date, description, value, units)
  - value is TEXT — ALWAYS cast: CAST(value AS FLOAT) before numeric comparison.
    Safe pattern for dirty data: CAST(NULLIF(value, '') AS FLOAT)
  - KEY OBSERVATIONS in Synthea (description and units):
      HbA1c     : description ILIKE '%Hemoglobin A1c%'        units='%'
                  If filtering by value, you MUST JOIN observations o ON p.id = o.patient_id.
                  Example: CAST(o.value AS FLOAT) > 7.0 (Adapt the number to the user's request).
      Systolic BP: description ILIKE '%Systolic Blood Pressure%'  units='mm[Hg]'
      Diastolic BP: description ILIKE '%Diastolic Blood Pressure%' units='mm[Hg]'
      Glucose   : description ILIKE '%Glucose%'               units='mg/dL'
      BMI       : description ILIKE '%Body Mass Index%'       units='kg/m2'
      Cholesterol: description ILIKE '%Total Cholesterol%'    units='mg/dL'
      eGFR      : description ILIKE '%Glomerular%'            units='mL/min/{1.73_m2}'
      Creatinine: description ILIKE '%Creatinine%'            units='mg/dL'
  - DATA IS HISTORICAL. NEVER use CURRENT_DATE for observation windows. Anchor to:
    (SELECT MAX(obs_date) FROM observations) - INTERVAL '12 months'
  - Most recent value per patient:
    SELECT DISTINCT ON (patient_id) patient_id, CAST(value AS FLOAT) AS val, obs_date
    FROM observations WHERE description ILIKE '%HbA1c%'
    ORDER BY patient_id, obs_date DESC

clinical_notes(id, patient_id, encounter_id, note_type, content, redacted_content, created_at)
documents(id, source_name, doc_type, chunk_index, content, ingested_at)
pro_responses(id, patient_id, instrument, score_mobility, score_self_care, score_activities,
              score_pain, score_anxiety, eq5d_index, recorded_at)
operational_metrics(id, trust_name, metric_name, metric_value, period_start, period_end)
ehg_records(id, record_name, gestation_weeks, preterm, rectime_minutes, maternal_age, parity,
            abortions, weight_kg, hypertension, diabetes, placental_position,
            bleeding_first_trimester, bleeding_second_trimester, funneling, smoker,
            n_signals, sample_rate, n_samples)
ehg_features(id, record_name, channel, variant, sample_rate, n_samples, mean, std, rms,
             p2p, skew, kurtosis, bandpower_0_08_0_3, bandpower_0_3_1_0,
             bandpower_1_0_3_0, bandpower_3_0_4_0)
mimic_patients(id, subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag)
mimic_admissions(id, hadm_id, subject_id, admittime, dischtime, deathtime, admission_type,
                 admission_location, discharge_location, insurance, language, religion,
                 marital_status, ethnicity, edregtime, edouttime, diagnosis,
                 hospital_expire_flag, has_chartevents_data)
mimic_icustays(id, icustay_id, subject_id, hadm_id, dbsource, first_careunit, last_careunit,
               first_wardid, last_wardid, intime, outtime, los)
mimic_d_icd_diagnoses(id, icd9_code, short_title, long_title)
mimic_diagnoses_icd(id, row_id, subject_id, hadm_id, seq_num, icd9_code)
mimic_d_labitems(id, itemid, label, fluid, category, loinc_code)
mimic_labevents(id, row_id, subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, flag)
mimic_prescriptions(id, row_id, subject_id, hadm_id, icustay_id, startdate, enddate, drug_type, drug,
                    drug_name_poe, drug_name_generic, formulary_drug_cd, gsn, ndc, prod_strength,
                    dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, route)
mimic_admission_features(id, hadm_id, subject_id, age_at_admit, icu_stay_count, icu_los_days,
                         has_diabetes, has_hypertension, hospital_expire_flag, diagnosis_text)

Relationships:
- encounters.patient_id → patients.id
- conditions.patient_id → patients.id; conditions.encounter_id → encounters.id
- medications.patient_id → patients.id
- observations.patient_id → patients.id
- clinical_notes.patient_id → patients.id
- pro_responses.patient_id → patients.id
- ehg_features.record_name → ehg_records.record_name
- mimic_admissions.subject_id → mimic_patients.subject_id
- mimic_icustays.hadm_id → mimic_admissions.hadm_id
- mimic_diagnoses_icd.icd9_code → mimic_d_icd_diagnoses.icd9_code
- mimic_labevents.itemid → mimic_d_labitems.itemid

MIMIC ICD-9 PATTERNS:
  - mimic_diagnoses_icd stores raw ICD-9 codes; join with mimic_d_icd_diagnoses for readable labels.
  - ALWAYS prefer mimic_admission_features for common cohort questions — it has pre-joined boolean flags:
      has_diabetes, has_hypertension, hospital_expire_flag, icu_los_days, age_at_admit
  - ICD-9 filter example — diabetic admissions (codes start with 250):
      SELECT ma.hadm_id, ma.diagnosis
      FROM mimic_diagnoses_icd md
      JOIN mimic_admissions ma ON md.hadm_id = ma.hadm_id
      WHERE md.icd9_code LIKE '250%'
  - mimic_admission_features shortcut (preferred):
      SELECT * FROM mimic_admission_features WHERE has_diabetes = TRUE
  - Common ICD-9 prefixes: 250=diabetes, 401=hypertension, 410-414=cardiac, 490-496=respiratory

Query rules (strictly follow):
1. LIMIT 100 on all SELECT queries unless they use COUNT/SUM/AVG/MIN/MAX/GROUP BY.
2. Use ILIKE for all text searches.
3. CAST(value AS FLOAT) before any numeric comparison on observations.value.
4. Use condition_description ILIKE — never icd10_code for filtering conditions.
5. Use DISTINCT ON (patient_id) ORDER BY patient_id, obs_date DESC for most-recent-per-patient.
6. ONLY use historical time windows (MAX(obs_date) - INTERVAL) if the user EXPLICITLY asks for "recent", "last year", or "current". Otherwise, do not filter by date.
7. Prefer mimic_admission_features for MIMIC cohort/outcome questions before joining raw MIMIC tables.
8. In PostgreSQL, ROUND(value, N) requires NUMERIC. ALWAYS wrap with CAST: ROUND(CAST(expression AS NUMERIC), 2). NEVER call ROUND(avg_col, 2) directly on AVG() or other float expressions — this causes a type error. Correct pattern: ROUND(CAST(AVG(col) AS NUMERIC), 2).
9. ALWAYS select the specific columns relevant to the user's question alongside the ID. Ensure you explicitly JOIN every table you select from or filter on (e.g. JOIN observations o ON p.id = o.patient_id).
10. For exclusion logic (e.g. "not on medication"), ALWAYS use NOT EXISTS or LEFT JOIN ... IS NULL. NEVER use INNER JOIN with != or IS FALSE.
"""

SQL_SYSTEM_PROMPT = f"""You are a PostgreSQL query expert for a healthcare database.
Generate a single valid PostgreSQL query to answer the user's question.
Use only the tables and columns from the schema below.
Always include LIMIT 100 unless the query is a COUNT or aggregation.
Return ONLY the SQL query, nothing else — no explanation, no markdown fences.

{SCHEMA_DESCRIPTION}"""

SQL_NOLIMIT_SYSTEM_PROMPT = f"""You are a PostgreSQL query expert for a healthcare database.
Generate a single valid PostgreSQL query to answer the user's question.
Use only the tables and columns from the schema below.
Do NOT include any LIMIT clause anywhere in the query — all rows are required.
Do NOT include LIMIT in subqueries or CTEs either.
Return ONLY the SQL query, nothing else — no explanation, no markdown fences.

{SCHEMA_DESCRIPTION}"""


def _deterministic_sql_for_query(query: str) -> str | None:
    lower = (query or "").lower()
    
    if "uncontrolled diabetes" in lower and "not on" in lower and "medication" in lower:
        return """
SELECT DISTINCT p.id, p.synthea_id, p.gender, p.birth_date, CAST(o.value AS FLOAT) as hba1c
FROM patients p
JOIN conditions c ON p.id = c.patient_id
JOIN observations o ON p.id = o.patient_id
WHERE c.condition_description ILIKE '%diabetes%'
  AND o.description ILIKE '%Hemoglobin A1c%'
  AND CAST(o.value AS FLOAT) > 8
  AND NOT EXISTS (
      SELECT 1 
      FROM medications m 
      WHERE m.patient_id = p.id 
      AND (m.drug_name ILIKE '%metformin%' OR m.drug_name ILIKE '%insulin%' OR m.drug_name ILIKE '%glipizide%' OR m.drug_name ILIKE '%glyburide%')
  )
LIMIT 100
"""

    if all(token in lower for token in ("ace inhibitor", "nsaid", "over 65")) and (
        "ckd" in lower or "renal insufficiency" in lower or "kidney" in lower or "renal risk" in lower
    ):
        return """
WITH older_patients AS (
    SELECT id AS patient_id
    FROM patients
    WHERE DATE_PART('year', AGE(birth_date)) > 65
),
ace_rx AS (
    SELECT patient_id, start_date, stop_date
    FROM medications
    WHERE drug_name ILIKE '%lisinopril%'
       OR drug_name ILIKE '%ramipril%'
       OR drug_name ILIKE '%enalapril%'
       OR drug_name ILIKE '%perindopril%'
       OR drug_name ILIKE '%captopril%'
       OR drug_name ILIKE '%benazepril%'
       OR drug_name ILIKE '%quinapril%'
       OR drug_name ILIKE '%fosinopril%'
),
nsaid_rx AS (
    SELECT patient_id, start_date, stop_date
    FROM medications
    WHERE drug_name ILIKE '%ibuprofen%'
       OR drug_name ILIKE '%naproxen%'
       OR drug_name ILIKE '%diclofenac%'
       OR drug_name ILIKE '%meloxicam%'
       OR drug_name ILIKE '%indomethacin%'
       OR drug_name ILIKE '%celecoxib%'
       OR drug_name ILIKE '%ketorolac%'
),
concurrent_patients AS (
    SELECT DISTINCT o.patient_id
    FROM older_patients o
    JOIN ace_rx a ON a.patient_id = o.patient_id
    JOIN nsaid_rx n ON n.patient_id = o.patient_id
    WHERE a.start_date <= COALESCE(n.stop_date, a.start_date + INTERVAL '1 day')
      AND n.start_date <= COALESCE(a.stop_date, n.start_date + INTERVAL '1 day')
),
renal_patients AS (
    SELECT DISTINCT patient_id
    FROM conditions
    WHERE condition_description ILIKE '%chronic kidney disease%'
       OR condition_description ILIKE '%renal insufficiency%'
       OR condition_description ILIKE '%kidney failure%'
)
SELECT
    COUNT(*) AS patients_over_65_with_ace_and_nsaid,
    COUNT(r.patient_id) AS with_ckd_or_renal_insufficiency
FROM concurrent_patients c
LEFT JOIN renal_patients r ON r.patient_id = c.patient_id
"""
    return None


async def generate_and_run_sql(query: str, conn: asyncpg.Connection) -> dict:
    sql = _deterministic_sql_for_query(query) or await _generate_sql(query)
    if _looks_truncated_sql(sql):
        logger.warning("Generated SQL looked truncated; retrying with higher token budget")
        sql = await _generate_sql(query, token_budget=1200)
    sql = _ensure_safe_limit(sql)

    result = await _execute_sql(conn, sql)
    if result.get("error") and result["error"]:
        logger.warning("SQL failed (%s); retrying with error context", result["error"])
        sql = await _generate_sql(query, error_context=result["error"])
        if _looks_truncated_sql(sql):
            sql = await _generate_sql(query, error_context=result["error"], token_budget=1200)
        sql = _ensure_safe_limit(sql)
        result = await _execute_sql(conn, sql)

    # If we got zero rows without an error, try one corrective retry. This commonly happens
    # when the LLM picks an incorrect join path (returns 0 rows) but the SQL is valid.
    if (not result.get("error")) and result.get("row_count", 0) == 0:
        sql2 = await _generate_sql(
            query,
            error_context=(
                "Previous query returned 0 rows (no SQL error). Common Synthea format issues: "
                "1) HbA1c units are '%' not 'mmol/mol' — use CAST(value AS FLOAT) > 7.0 not > 53; "
                "2) icd10_code stores SNOMED CT codes — use condition_description ILIKE '%keyword%' instead; "
                "3) Date windows: anchor to MAX(obs_date)/MAX(start_date), never CURRENT_DATE; "
                "4) Drug names need ILIKE fuzzy match. Rewrite fixing the likely format mismatch."
            ),
        )
        if _looks_truncated_sql(sql2):
            sql2 = await _generate_sql(
                query,
                error_context=(
                    "Previous query returned 0 rows and the generated SQL may have been truncated. "
                    "Rewrite the full PostgreSQL query completely and return only the final SQL."
                ),
                token_budget=1200,
            )
        sql = _ensure_safe_limit(sql2)
        result = await _execute_sql(conn, sql)

    result["sql"] = sql
    return result


def _ensure_safe_limit(sql: str, skip_limit: bool = False) -> str:
    """Inject LIMIT 100 on raw SELECT queries to prevent memory blow-ups.

    Pass skip_limit=True only for queries where the full result set is needed
    (e.g. for statistical sampling before a local model run).  In practice,
    predictive-modeling steps go through predictive_engine.py and bypass this
    function entirely — the flag exists purely as a safety escape hatch.
    """
    import re

    if skip_limit:
        return (sql or "").strip().rstrip(";").strip()

    s = (sql or "").strip().rstrip(";").strip()
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return s
    if re.search(r"\blimit\b", low):
        return s
    aggregation_markers = ("count(", "group by", "sum(", "avg(", "min(", "max(")
    if any(m in low for m in aggregation_markers):
        return s
    return s + " LIMIT 100"


def _looks_truncated_sql(sql: str) -> bool:
    text = (sql or "").strip()
    if not text:
        return True
    if text.count("(") != text.count(")"):
        return True
    lowered = text.lower()
    if lowered.endswith(",") or lowered.endswith("from") or lowered.endswith("where") or lowered.endswith("and"):
        return True
    return False


async def _generate_sql(query: str, error_context: str = None, token_budget: int = 1500) -> str:
    user_msg = query
    if error_context:
        user_msg = f"{query}\n\nPrevious attempt failed with error: {error_context}\nPlease fix the SQL."

    sql = await chat(
        model=MAIN_MODEL,
        ollama_model=OLLAMA_MAIN_MODEL,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=token_budget,
        temperature=0.0,
    )
    # Strip markdown fences
    if sql.startswith("```"):
        parts = sql.split("```")
        sql = parts[1] if len(parts) > 1 else sql
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    return sql.strip()


async def _generate_sql_nolimit(query: str, token_budget: int = 1500) -> str:
    """Generate SQL explicitly without any LIMIT clause.

    Used for predictive/modeling queries where the full dataset is required.
    Uses SQL_NOLIMIT_SYSTEM_PROMPT which instructs the model to never add LIMIT
    anywhere — in the outer query, subqueries, or CTEs.
    The regex strip in predictive_engine then acts as a safety net only.
    """
    sql = await chat(
        model=MAIN_MODEL,
        ollama_model=OLLAMA_MAIN_MODEL,
        messages=[
            {"role": "system", "content": SQL_NOLIMIT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=token_budget,
        temperature=0.0,
    )
    if sql.startswith("```"):
        parts = sql.split("```")
        sql = parts[1] if len(parts) > 1 else sql
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    return sql.strip()


async def _execute_sql(conn: asyncpg.Connection, sql: str) -> dict:
    try:
        started = time.perf_counter()
        rows = await conn.fetch(sql, timeout=30)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        results = [dict(r) for r in rows]
        # Convert non-serialisable types
        for row in results:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
        logger.info("SQL executed in %dms; rows=%d", elapsed_ms, len(results))
        return {"results": results, "row_count": len(results), "error": None}
    except Exception as e:
        logger.error("SQL execution error: %s\nSQL: %s", e, sql)
        return {"results": [], "row_count": 0, "error": str(e)}
