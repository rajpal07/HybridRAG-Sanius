"""
Dynamic Predictive Modeling Engine
====================================
Key design decisions:
  1. SQL is ALWAYS generated fresh via the SQL agent (never trusted from the
     LLM planner's JSON, which can be truncated by small models).
  2. The full table is fetched without LIMIT — large datasets stay in Python,
     never in the LLM context window.
  3. scikit-learn runs locally: training a Random Forest on 10k rows takes ~1s.
"""
from __future__ import annotations

import logging
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning(
        "scikit-learn not installed in this environment. "
        "Run: pip install scikit-learn  (inside your .venv)"
    )


import re


def _build_sql_question(step: dict[str, Any]) -> str:
    """
    Build a precise natural-language question for the SQL agent.
    The planner provides the intent; the SQL agent — which has the full schema
    description and retry logic — decides the exact query.
    """
    target = (step.get("target_column") or "hospital_expire_flag").strip()
    question = (step.get("question") or "").strip()

    base = question or (
        f"Select all available clinical feature columns and the target column "
        f"'{target}' from the most relevant table for this analysis."
    )
    return (
        f"{base}\n\n"
        f"IMPORTANT: Select ALL feature columns and '{target}'. "
        f"Do NOT add a LIMIT clause — all rows are required for statistical modeling."
    )


async def _resolve_sql(step: dict[str, Any]) -> str:
    """
    Dynamically generate the SQL for a predictive step by delegating entirely
    to the SQL agent. The SQL agent knows the full database schema and has
    built-in retry and error-correction — making this robust for any table
    or target column, with no hardcoded templates needed.

    After generation, the LIMIT clause is stripped since we need all rows.
    """
    from backend.agents.sql_agent import _generate_sql_nolimit  # noqa: PLC0415

    question = _build_sql_question(step)
    logger.info("Predictive step: delegating SQL generation to SQL agent (no-limit prompt).")

    sql = await _generate_sql_nolimit(question)

    # Strip any LIMIT the SQL agent may have injected (safe default for row queries)
    sql = re.sub(r"\bLIMIT\s+\d+\b", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip(";")

    # Validate basic structure
    lower = sql.lower()
    if not lower.startswith(("select", "with")):
        raise ValueError(f"SQL agent returned a non-SELECT statement: {sql[:120]}")
    if sql.count("(") != sql.count(")"):
        raise ValueError(f"Generated SQL has unbalanced parentheses (likely truncated): {sql[:120]}")

    return sql


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_predictive_step(
    conn: asyncpg.Connection,
    step: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a predictive_modeling step end-to-end.

    step keys used:
      name          : label for logging
      sql           : optional SQL from planner (validated before use)
      question      : optional NL question (used to generate SQL if needed)
      target_column : column to predict (required)
    """
    step_name = step.get("name", "predictive_step")
    target_col = (step.get("target_column") or "hospital_expire_flag").strip()

    if not _SKLEARN_AVAILABLE:
        return _error_result(
            step_name,
            "scikit-learn is not installed in this Python environment. "
            "Activate your .venv and run: pip install scikit-learn",
        )

    # --- 1. Resolve & execute SQL ---
    try:
        sql = await _resolve_sql(step)
    except Exception as exc:
        return _error_result(step_name, f"SQL resolution failed: {exc}")

    try:
        rows = await conn.fetch(sql, timeout=120)
        if not rows:
            return _error_result(step_name, "Query returned 0 rows.", sql=sql)
        df = pd.DataFrame([dict(r) for r in rows])
        logger.info("Predictive step '%s': fetched %d rows, %d columns", step_name, len(df), len(df.columns))
    except Exception as exc:
        return _error_result(step_name, f"SQL execution failed: {exc}", sql=sql)

    # Ensure target column exists after resolving column names
    if target_col not in df.columns:
        # Try case-insensitive match
        col_map = {c.lower(): c for c in df.columns}
        if target_col.lower() in col_map:
            target_col = col_map[target_col.lower()]
        else:
            return _error_result(
                step_name,
                f"Target column '{target_col}' not in results. Available: {list(df.columns)}",
                sql=sql,
            )

    # --- 2. Preprocess ---
    df = df.copy()
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    X = _drop_id_columns(X)
    X, _ = _encode_features(X)
    is_classifier, y = _encode_target(y_raw)

    if X.shape[1] == 0:
        return _error_result(step_name, "No usable feature columns remain after preprocessing.", sql=sql)
    if len(y) < 20:
        return _error_result(step_name, f"Too few rows ({len(y)}) to train a reliable model.", sql=sql)

    # --- 3. Train ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if is_classifier:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=8, n_jobs=-1,
                random_state=42, class_weight="balanced",
            )
            model.fit(X_train, y_train)
            metric_name, metric_value = "accuracy", float(accuracy_score(y_test, model.predict(X_test)))
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=8, n_jobs=-1, random_state=42,
            )
            model.fit(X_train, y_train)
            metric_name, metric_value = "r2", float(r2_score(y_test, model.predict(X_test)))

        logger.info(
            "Predictive step '%s': trained on %d rows — %s=%.3f",
            step_name, len(y), metric_name, metric_value,
        )
    except Exception as exc:
        return _error_result(step_name, f"Model training failed: {exc}", sql=sql)

    # --- 4. Feature importances ---
    feature_importances = [
        {"feature": col, "importance": round(float(imp), 4)}
        for col, imp in sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
    ]

    return {
        "name": step_name,
        "type": "predictive_modeling",
        "sql": sql,
        "target_column": target_col,
        "model_type": "classifier" if is_classifier else "regressor",
        "metric_name": metric_name,
        "metric_value": round(metric_value, 4),
        "row_count": len(y),
        "feature_importances": feature_importances[:10],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(name: str, msg: str, sql: str | None = None) -> dict[str, Any]:
    logger.warning("Predictive step '%s' error: %s", name, msg)
    return {
        "name": name,
        "type": "predictive_modeling",
        "sql": sql,
        "target_column": None,
        "model_type": None,
        "metric_name": None,
        "metric_value": None,
        "row_count": 0,
        "feature_importances": [],
        "error": msg,
    }


def _drop_id_columns(X: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"id", "row_id", "hadm_id", "subject_id", "icustay_id"}
    drop = [c for c in X.columns if c.lower() in id_cols]
    return X.drop(columns=drop, errors="ignore")


def _encode_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = X.copy()
    encoding_map: dict[str, Any] = {}
    for col in list(result.columns):
        dtype = result[col].dtype
        if str(dtype) in ("object", "category"):
            non_null = result[col].dropna().astype(str)
            if non_null.empty:
                result.drop(columns=[col], inplace=True)
                continue
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].fillna("__missing__").astype(str))
            encoding_map[col] = le
        elif str(dtype) in ("bool", "boolean"):
            result[col] = result[col].fillna(False).astype(int)
        else:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result, encoding_map


def _encode_target(y_raw: pd.Series) -> tuple[bool, np.ndarray]:
    y = y_raw.dropna()
    if str(y.dtype) in ("bool", "boolean"):
        return True, y.astype(int).values
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
        return True, y.values
    if y.dtype == object:
        return True, LabelEncoder().fit_transform(y.astype(str))
    return False, pd.to_numeric(y, errors="coerce").fillna(0).values
