from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

from backend.config import EHG_SOURCE_DIR
from backend.database import get_pool

logger = logging.getLogger(__name__)


def _to_bool(v: str | None) -> bool | None:
    if v is None:
        return None
    s = v.strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


def _to_int(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v.strip()))
    except Exception:
        return None


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(v.strip())
    except Exception:
        return None


def _find_dataset_root() -> Path:
    # Prefer explicit env var; otherwise try the user-provided default path.
    if EHG_SOURCE_DIR:
        p = Path(EHG_SOURCE_DIR)
        if p.exists():
            return p

    raise FileNotFoundError(
        "EHG dataset not found. Set EHG_SOURCE_DIR to the "
        "term-preterm-ehg-database-1.0.1 folder."
    )


def _parse_header(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty header: {path}")

    first = lines[0].strip().split()
    record_name = first[0] if first else path.stem
    n_signals = _to_int(first[1]) if len(first) > 1 else None
    sample_rate = _to_float(first[2]) if len(first) > 2 else None
    n_samples = _to_int(first[3]) if len(first) > 3 else None

    meta: dict[str, str] = {}
    for ln in lines:
        if not ln.lstrip().startswith("#"):
            continue
        s = ln.lstrip("#").strip()
        if not s or s.lower() == "comments:":
            continue
        parts = s.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0].strip(), parts[1].strip()
        meta[key] = value

    rec_id = _to_int(meta.get("RecID"))
    gestation_weeks = _to_float(meta.get("Gestation"))
    preterm = (gestation_weeks < 37.0) if gestation_weeks is not None else None

    return {
        "record_name": record_name,
        "rec_id": rec_id,
        "gestation_weeks": gestation_weeks,
        "preterm": preterm,
        "rectime_minutes": _to_float(meta.get("Rectime")),
        "maternal_age": _to_int(meta.get("Age")),
        "parity": _to_int(meta.get("Parity")),
        "abortions": _to_int(meta.get("Abortions")),
        "weight_kg": _to_float(meta.get("Weight")),
        "hypertension": _to_bool(meta.get("Hypertension")),
        "diabetes": _to_bool(meta.get("Diabetes")),
        "placental_position": meta.get("Placental_position"),
        "bleeding_first_trimester": _to_bool(meta.get("Bleeding_first_trimester")),
        "bleeding_second_trimester": _to_bool(meta.get("Bleeding_second_trimester")),
        "funneling": meta.get("Funneling"),
        "smoker": _to_bool(meta.get("Smoker")),
        "n_signals": n_signals,
        "sample_rate": sample_rate,
        "n_samples": n_samples,
    }


def _parse_signal_lines(header_path: Path) -> list[dict]:
    # Parse signal lines from .hea. The first line is record metadata.
    lines = header_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[dict] = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            break
        parts = s.split()
        if len(parts) < 2:
            continue
        # Example: tpehg1007.dat 16 13107/mV 16 0 1093 -7439 0 1_DOCFILT-4-0.08-4
        fname = parts[0]
        fmt = parts[1]
        gain_uom = parts[2] if len(parts) > 2 else ""
        gain = None
        uom = None
        if "/" in gain_uom:
            g, u = gain_uom.split("/", 1)
            try:
                gain = float(g)
            except Exception:
                gain = None
            uom = u
        label = parts[-1] if parts else ""
        channel = None
        variant = "raw"
        if "_" in label:
            # "1_DOCFILT-..." => channel 1, variant DOCFILT-...
            ch, var = label.split("_", 1)
            channel = _to_int(ch)
            variant = var.strip() if var else "raw"
        else:
            channel = _to_int(label) if label else None
            variant = "raw"
        out.append(
            {
                "data_file": fname,
                "fmt": fmt,
                "gain": gain,
                "uom": uom,
                "channel": channel,
                "variant": variant,
            }
        )
    return out


def _read_wfdb_16(dat_path: Path, n_signals: int, n_samples: int) -> np.ndarray:
    # WFDB format "16" in this dataset is 16-bit signed integers, multiplexed by sample.
    raw = np.fromfile(dat_path, dtype="<i2")
    expected = int(n_signals) * int(n_samples)
    if raw.size < expected:
        raise ValueError(f"Short .dat: expected {expected} int16, got {raw.size} ({dat_path})")
    raw = raw[:expected]
    return raw.reshape((n_samples, n_signals))


def _bandpower(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    if x.size < 8 or fs <= 0:
        return float("nan")
    f, pxx = welch(x, fs=fs, nperseg=min(2048, x.size))
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(pxx[mask], f[mask]))


def _compute_features(x: np.ndarray, fs: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    mean_v = float(np.mean(x))
    std_v = float(np.std(x))
    rms_v = float(np.sqrt(np.mean(x * x)))
    p2p_v = float(np.max(x) - np.min(x))
    sk = float(skew(x)) if x.size > 8 else float("nan")
    ku = float(kurtosis(x)) if x.size > 8 else float("nan")

    return {
        "mean": mean_v,
        "std": std_v,
        "rms": rms_v,
        "p2p": p2p_v,
        "skew": sk,
        "kurtosis": ku,
        "bandpower_0_08_0_3": _bandpower(x, fs, 0.08, 0.3),
        "bandpower_0_3_1_0": _bandpower(x, fs, 0.3, 1.0),
        "bandpower_1_0_3_0": _bandpower(x, fs, 1.0, 3.0),
        "bandpower_3_0_4_0": _bandpower(x, fs, 3.0, 4.0),
    }


async def ingest_ehg() -> dict:
    root = _find_dataset_root()
    records_file = root / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"Missing RECORDS file at {records_file}")

    record_rel_paths = [
        ln.strip()
        for ln in records_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    ]

    base_dir = root
    inserted = 0
    updated = 0
    features_upserted = 0

    pool = await get_pool()
    async with pool.acquire() as conn:
        for rel in record_rel_paths:
            rec = (base_dir / rel).as_posix()
            hea_path = base_dir / f"{rel}.hea"
            dat_path = base_dir / f"{rel}.dat"
            if not hea_path.exists():
                logger.warning("Skipping missing header: %s", hea_path)
                continue

            row = _parse_header(hea_path)
            sig_lines = _parse_signal_lines(hea_path)
            row["header_path"] = str(hea_path)
            row["data_path"] = str(dat_path) if dat_path.exists() else None

            res = await conn.execute(
                """
                INSERT INTO ehg_records (
                    record_name, rec_id, gestation_weeks, preterm, rectime_minutes,
                    maternal_age, parity, abortions, weight_kg,
                    hypertension, diabetes, placental_position,
                    bleeding_first_trimester, bleeding_second_trimester,
                    funneling, smoker,
                    n_signals, sample_rate, n_samples,
                    header_path, data_path
                )
                VALUES (
                    $1,$2,$3,$4,$5,
                    $6,$7,$8,$9,
                    $10,$11,$12,
                    $13,$14,
                    $15,$16,
                    $17,$18,$19,
                    $20,$21
                )
                ON CONFLICT (record_name) DO UPDATE SET
                    rec_id=EXCLUDED.rec_id,
                    gestation_weeks=EXCLUDED.gestation_weeks,
                    preterm=EXCLUDED.preterm,
                    rectime_minutes=EXCLUDED.rectime_minutes,
                    maternal_age=EXCLUDED.maternal_age,
                    parity=EXCLUDED.parity,
                    abortions=EXCLUDED.abortions,
                    weight_kg=EXCLUDED.weight_kg,
                    hypertension=EXCLUDED.hypertension,
                    diabetes=EXCLUDED.diabetes,
                    placental_position=EXCLUDED.placental_position,
                    bleeding_first_trimester=EXCLUDED.bleeding_first_trimester,
                    bleeding_second_trimester=EXCLUDED.bleeding_second_trimester,
                    funneling=EXCLUDED.funneling,
                    smoker=EXCLUDED.smoker,
                    n_signals=EXCLUDED.n_signals,
                    sample_rate=EXCLUDED.sample_rate,
                    n_samples=EXCLUDED.n_samples,
                    header_path=EXCLUDED.header_path,
                    data_path=EXCLUDED.data_path,
                    ingested_at=NOW()
                """,
                row["record_name"],
                row["rec_id"],
                row["gestation_weeks"],
                row["preterm"],
                row["rectime_minutes"],
                row["maternal_age"],
                row["parity"],
                row["abortions"],
                row["weight_kg"],
                row["hypertension"],
                row["diabetes"],
                row["placental_position"],
                row["bleeding_first_trimester"],
                row["bleeding_second_trimester"],
                row["funneling"],
                row["smoker"],
                row["n_signals"],
                row["sample_rate"],
                row["n_samples"],
                row["header_path"],
                row["data_path"],
            )
            # asyncpg returns "INSERT 0 1" for insert, "UPDATE 0 1" for update
            if res.startswith("INSERT"):
                inserted += 1
            else:
                updated += 1

            # Compute lightweight waveform features if the .dat exists.
            if not dat_path.exists():
                continue
            n_signals = row.get("n_signals") or 0
            n_samples = row.get("n_samples") or 0
            fs = row.get("sample_rate") or 0.0
            if not n_signals or not n_samples or not fs:
                continue

            try:
                mat = _read_wfdb_16(dat_path, int(n_signals), int(n_samples))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed reading %s: %s", dat_path, e)
                continue

            # Select the first three "raw" channels (1,2,3) plus their DOCFILT variants.
            for sig_idx, sig in enumerate(sig_lines, start=0):
                ch = sig.get("channel")
                var = sig.get("variant") or "raw"
                if ch not in (1, 2, 3):
                    continue
                if sig_idx >= mat.shape[1]:
                    continue
                x = mat[:, sig_idx]
                feats = _compute_features(x, float(fs))
                if not feats:
                    continue

                await conn.execute(
                    """
                    INSERT INTO ehg_features (
                        record_name, channel, variant, sample_rate, n_samples,
                        mean, std, rms, p2p, skew, kurtosis,
                        bandpower_0_08_0_3, bandpower_0_3_1_0, bandpower_1_0_3_0, bandpower_3_0_4_0
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                    ON CONFLICT (record_name, channel, variant) DO UPDATE SET
                        sample_rate=EXCLUDED.sample_rate,
                        n_samples=EXCLUDED.n_samples,
                        mean=EXCLUDED.mean,
                        std=EXCLUDED.std,
                        rms=EXCLUDED.rms,
                        p2p=EXCLUDED.p2p,
                        skew=EXCLUDED.skew,
                        kurtosis=EXCLUDED.kurtosis,
                        bandpower_0_08_0_3=EXCLUDED.bandpower_0_08_0_3,
                        bandpower_0_3_1_0=EXCLUDED.bandpower_0_3_1_0,
                        bandpower_1_0_3_0=EXCLUDED.bandpower_1_0_3_0,
                        bandpower_3_0_4_0=EXCLUDED.bandpower_3_0_4_0,
                        computed_at=NOW()
                    """,
                    row["record_name"],
                    int(ch),
                    str(var),
                    float(fs),
                    int(n_samples),
                    feats["mean"],
                    feats["std"],
                    feats["rms"],
                    feats["p2p"],
                    feats["skew"],
                    feats["kurtosis"],
                    feats["bandpower_0_08_0_3"],
                    feats["bandpower_0_3_1_0"],
                    feats["bandpower_1_0_3_0"],
                    feats["bandpower_3_0_4_0"],
                )
                features_upserted += 1

    return {
        "dataset_root": str(root),
        "records_total": len(record_rel_paths),
        "inserted": inserted,
        "updated": updated,
        "features_upserted": features_upserted,
    }
