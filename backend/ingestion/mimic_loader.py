import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import asyncpg

from backend.config import MIMIC_SOURCE_DIR
from backend.database import get_pool

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def _path(name: str) -> Path:
    return Path(MIMIC_SOURCE_DIR) / name


def _safe_ts(value):
    if pd.isna(value) or value is None:
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def _safe_str(value) -> Optional[str]:
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value) -> Optional[int]:
    if pd.isna(value) or value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value) -> Optional[float]:
    if pd.isna(value) or value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value) -> Optional[bool]:
    if pd.isna(value) or value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _age_at_admit(admittime, dob) -> Optional[float]:
    admit_ts = _safe_ts(admittime)
    dob_ts = _safe_ts(dob)
    if not admit_ts or not dob_ts:
        return None
    try:
        age = (admit_ts - dob_ts).total_seconds() / (365.25 * 24 * 3600)
    except Exception:
        return None
    if age < 0:
        return None
    if age > 120:
        return 90.0
    return round(age, 2)


async def _batch_insert(conn: asyncpg.Connection, query: str, rows: list[tuple]):
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start : start + BATCH_SIZE]
        await conn.executemany(query, chunk)
        logger.info("  inserted %d / %d rows", min(start + BATCH_SIZE, len(rows)), len(rows))


def _read_csv(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(f"MIMIC file not found: {path}")
    return pd.read_csv(path, low_memory=False, usecols=usecols)


async def load_mimic_patients(conn: asyncpg.Connection):
    df = _read_csv("PATIENTS.csv")
    rows = [
        (
            _safe_int(r.subject_id),
            _safe_str(r.gender),
            _safe_ts(r.dob),
            _safe_ts(r.dod),
            _safe_ts(r.dod_hosp),
            _safe_ts(r.dod_ssn),
            _safe_bool(r.expire_flag),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_patients
            (subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (subject_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            dob = EXCLUDED.dob,
            dod = EXCLUDED.dod,
            dod_hosp = EXCLUDED.dod_hosp,
            dod_ssn = EXCLUDED.dod_ssn,
            expire_flag = EXCLUDED.expire_flag
        """,
        rows,
    )


async def load_mimic_admissions(conn: asyncpg.Connection):
    df = _read_csv("ADMISSIONS.csv")
    rows = [
        (
            _safe_int(r.hadm_id),
            _safe_int(r.subject_id),
            _safe_ts(r.admittime),
            _safe_ts(r.dischtime),
            _safe_ts(r.deathtime),
            _safe_str(r.admission_type),
            _safe_str(r.admission_location),
            _safe_str(r.discharge_location),
            _safe_str(r.insurance),
            _safe_str(r.language),
            _safe_str(r.religion),
            _safe_str(r.marital_status),
            _safe_str(r.ethnicity),
            _safe_ts(r.edregtime),
            _safe_ts(r.edouttime),
            _safe_str(r.diagnosis),
            _safe_bool(r.hospital_expire_flag),
            _safe_bool(r.has_chartevents_data),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_admissions
            (hadm_id, subject_id, admittime, dischtime, deathtime, admission_type,
             admission_location, discharge_location, insurance, language, religion,
             marital_status, ethnicity, edregtime, edouttime, diagnosis,
             hospital_expire_flag, has_chartevents_data)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
        ON CONFLICT (hadm_id) DO UPDATE SET
            subject_id = EXCLUDED.subject_id,
            admittime = EXCLUDED.admittime,
            dischtime = EXCLUDED.dischtime,
            deathtime = EXCLUDED.deathtime,
            admission_type = EXCLUDED.admission_type,
            admission_location = EXCLUDED.admission_location,
            discharge_location = EXCLUDED.discharge_location,
            insurance = EXCLUDED.insurance,
            language = EXCLUDED.language,
            religion = EXCLUDED.religion,
            marital_status = EXCLUDED.marital_status,
            ethnicity = EXCLUDED.ethnicity,
            edregtime = EXCLUDED.edregtime,
            edouttime = EXCLUDED.edouttime,
            diagnosis = EXCLUDED.diagnosis,
            hospital_expire_flag = EXCLUDED.hospital_expire_flag,
            has_chartevents_data = EXCLUDED.has_chartevents_data
        """,
        rows,
    )


async def load_mimic_icustays(conn: asyncpg.Connection):
    df = _read_csv("ICUSTAYS.csv")
    rows = [
        (
            _safe_int(r.icustay_id),
            _safe_int(r.subject_id),
            _safe_int(r.hadm_id),
            _safe_str(r.dbsource),
            _safe_str(r.first_careunit),
            _safe_str(r.last_careunit),
            _safe_int(r.first_wardid),
            _safe_int(r.last_wardid),
            _safe_ts(r.intime),
            _safe_ts(r.outtime),
            _safe_float(r.los),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_icustays
            (icustay_id, subject_id, hadm_id, dbsource, first_careunit, last_careunit,
             first_wardid, last_wardid, intime, outtime, los)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        ON CONFLICT (icustay_id) DO UPDATE SET
            subject_id = EXCLUDED.subject_id,
            hadm_id = EXCLUDED.hadm_id,
            dbsource = EXCLUDED.dbsource,
            first_careunit = EXCLUDED.first_careunit,
            last_careunit = EXCLUDED.last_careunit,
            first_wardid = EXCLUDED.first_wardid,
            last_wardid = EXCLUDED.last_wardid,
            intime = EXCLUDED.intime,
            outtime = EXCLUDED.outtime,
            los = EXCLUDED.los
        """,
        rows,
    )


async def load_mimic_d_icd_diagnoses(conn: asyncpg.Connection):
    df = _read_csv("D_ICD_DIAGNOSES.csv")
    rows = [
        (_safe_str(r.icd9_code), _safe_str(r.short_title), _safe_str(r.long_title))
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_d_icd_diagnoses (icd9_code, short_title, long_title)
        VALUES ($1,$2,$3)
        ON CONFLICT (icd9_code) DO UPDATE SET
            short_title = EXCLUDED.short_title,
            long_title = EXCLUDED.long_title
        """,
        rows,
    )


async def load_mimic_diagnoses_icd(conn: asyncpg.Connection):
    df = _read_csv("DIAGNOSES_ICD.csv")
    rows = [
        (
            _safe_int(r.row_id),
            _safe_int(r.subject_id),
            _safe_int(r.hadm_id),
            _safe_int(r.seq_num),
            _safe_str(r.icd9_code),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_diagnoses_icd (row_id, subject_id, hadm_id, seq_num, icd9_code)
        VALUES ($1,$2,$3,$4,$5)
        ON CONFLICT (row_id) DO NOTHING
        """,
        rows,
    )


async def load_mimic_d_labitems(conn: asyncpg.Connection):
    df = _read_csv("D_LABITEMS.csv")
    rows = [
        (
            _safe_int(r.itemid),
            _safe_str(r.label),
            _safe_str(r.fluid),
            _safe_str(r.category),
            _safe_str(r.loinc_code),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_d_labitems (itemid, label, fluid, category, loinc_code)
        VALUES ($1,$2,$3,$4,$5)
        ON CONFLICT (itemid) DO UPDATE SET
            label = EXCLUDED.label,
            fluid = EXCLUDED.fluid,
            category = EXCLUDED.category,
            loinc_code = EXCLUDED.loinc_code
        """,
        rows,
    )


async def load_mimic_labevents(conn: asyncpg.Connection):
    df = _read_csv("LABEVENTS.csv")
    rows = [
        (
            _safe_int(r.row_id),
            _safe_int(r.subject_id),
            _safe_int(r.hadm_id),
            _safe_int(r.itemid),
            _safe_ts(r.charttime),
            _safe_str(r.value),
            _safe_float(r.valuenum),
            _safe_str(r.valueuom),
            _safe_str(r.flag),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_labevents
            (row_id, subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, flag)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (row_id) DO NOTHING
        """,
        rows,
    )


async def load_mimic_prescriptions(conn: asyncpg.Connection):
    df = _read_csv("PRESCRIPTIONS.csv")
    rows = [
        (
            _safe_int(r.row_id),
            _safe_int(r.subject_id),
            _safe_int(r.hadm_id),
            _safe_int(r.icustay_id),
            _safe_ts(r.startdate),
            _safe_ts(r.enddate),
            _safe_str(r.drug_type),
            _safe_str(r.drug),
            _safe_str(r.drug_name_poe),
            _safe_str(r.drug_name_generic),
            _safe_str(r.formulary_drug_cd),
            _safe_str(r.gsn),
            _safe_str(r.ndc),
            _safe_str(r.prod_strength),
            _safe_str(r.dose_val_rx),
            _safe_str(r.dose_unit_rx),
            _safe_str(r.form_val_disp),
            _safe_str(r.form_unit_disp),
            _safe_str(r.route),
        )
        for r in df.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_prescriptions
            (row_id, subject_id, hadm_id, icustay_id, startdate, enddate, drug_type, drug,
             drug_name_poe, drug_name_generic, formulary_drug_cd, gsn, ndc, prod_strength,
             dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, route)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
        ON CONFLICT (row_id) DO NOTHING
        """,
        rows,
    )


async def load_mimic_admission_features(conn: asyncpg.Connection):
    admissions = _read_csv("ADMISSIONS.csv", usecols=["subject_id", "hadm_id", "admittime", "diagnosis", "hospital_expire_flag"])
    patients = _read_csv("PATIENTS.csv", usecols=["subject_id", "dob"])
    icustays = _read_csv("ICUSTAYS.csv", usecols=["hadm_id", "los"])
    diagnoses = _read_csv("DIAGNOSES_ICD.csv", usecols=["hadm_id", "icd9_code"])
    d_icd = _read_csv("D_ICD_DIAGNOSES.csv", usecols=["icd9_code", "short_title", "long_title"])

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    merged = admissions.merge(patients, on="subject_id", how="left")
    merged["age_at_admit"] = merged.apply(
        lambda row: _age_at_admit(row["admittime"], row["dob"]),
        axis=1,
    )

    icu_agg = icustays.groupby("hadm_id", dropna=False).agg(
        icu_stay_count=("hadm_id", "size"),
        icu_los_days=("los", "sum"),
    )

    dx = diagnoses.merge(d_icd, on="icd9_code", how="left")
    dx["dx_text"] = (dx["short_title"].fillna("") + " " + dx["long_title"].fillna("")).str.lower()
    dx_agg = dx.groupby("hadm_id", dropna=False).agg(
        has_diabetes=("dx_text", lambda s: any("diabet" in value for value in s if isinstance(value, str))),
        has_hypertension=("dx_text", lambda s: any("hypertens" in value for value in s if isinstance(value, str))),
    )

    merged = merged.merge(icu_agg, on="hadm_id", how="left").merge(dx_agg, on="hadm_id", how="left")
    merged["icu_stay_count"] = merged["icu_stay_count"].fillna(0)
    merged["icu_los_days"] = merged["icu_los_days"].fillna(0.0)
    merged["has_diabetes"] = merged["has_diabetes"].fillna(False)
    merged["has_hypertension"] = merged["has_hypertension"].fillna(False)

    rows = [
        (
            _safe_int(r.hadm_id),
            _safe_int(r.subject_id),
            _safe_float(r.age_at_admit),
            _safe_int(r.icu_stay_count) or 0,
            _safe_float(r.icu_los_days) or 0.0,
            bool(r.has_diabetes),
            bool(r.has_hypertension),
            _safe_bool(r.hospital_expire_flag),
            _safe_str(r.diagnosis),
        )
        for r in merged.itertuples(index=False)
    ]
    await _batch_insert(
        conn,
        """
        INSERT INTO mimic_admission_features
            (hadm_id, subject_id, age_at_admit, icu_stay_count, icu_los_days,
             has_diabetes, has_hypertension, hospital_expire_flag, diagnosis_text)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (hadm_id) DO UPDATE SET
            subject_id = EXCLUDED.subject_id,
            age_at_admit = EXCLUDED.age_at_admit,
            icu_stay_count = EXCLUDED.icu_stay_count,
            icu_los_days = EXCLUDED.icu_los_days,
            has_diabetes = EXCLUDED.has_diabetes,
            has_hypertension = EXCLUDED.has_hypertension,
            hospital_expire_flag = EXCLUDED.hospital_expire_flag,
            diagnosis_text = EXCLUDED.diagnosis_text
        """,
        rows,
    )


async def ingest_mimic() -> dict:
    source_dir = Path(MIMIC_SOURCE_DIR)
    if not source_dir.exists():
        raise FileNotFoundError(f"MIMIC source directory not found: {source_dir}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        logger.info("=== Loading MIMIC-III demo data from %s ===", source_dir)
        await load_mimic_patients(conn)
        await load_mimic_admissions(conn)
        await load_mimic_icustays(conn)
        await load_mimic_d_icd_diagnoses(conn)
        await load_mimic_diagnoses_icd(conn)
        await load_mimic_d_labitems(conn)
        await load_mimic_labevents(conn)
        await load_mimic_prescriptions(conn)
        await load_mimic_admission_features(conn)

        counts = {}
        for table in [
            "mimic_patients",
            "mimic_admissions",
            "mimic_icustays",
            "mimic_diagnoses_icd",
            "mimic_labevents",
            "mimic_prescriptions",
            "mimic_admission_features",
        ]:
            counts[table] = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")

    logger.info("MIMIC ingestion complete: %s", counts)
    return counts
