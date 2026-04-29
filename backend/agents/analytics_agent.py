import asyncio
import logging
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

from backend.config import MAIN_MODEL, OLLAMA_MAIN_MODEL
from backend.agents.sql_agent import SCHEMA_DESCRIPTION, generate_and_run_sql
from backend.agents.predictive_engine import run_predictive_step
from backend.llm import chat_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner prompt — now fully understands predictive_modeling steps
# ---------------------------------------------------------------------------

ANALYTICS_PLAN_SYSTEM = f"""You are the planner in a multi-agent healthcare analytics system.
Your job is to decide whether a question is simple, complex, or requires predictive modeling,
and to emit a structured execution plan.

Return ONLY valid JSON with this schema:
{{
  "execution_mode": "simple" | "multi_step",
  "answerability": "direct" | "partial" | "insufficient",
  "reasoning": "short explanation",
  "steps": [
    {{
      "name": "short_step_name",
      "type": "sql" | "predictive_modeling",
      "question": "natural language task (used when type=sql)",
      "sql": "full SELECT query (used when type=predictive_modeling — NO LIMIT clause)",
      "target_column": "column to predict (required when type=predictive_modeling)",
      "must_succeed": true
    }}
  ],
  "missing_requirements": []
}}

Rules:
- Use type="sql" for aggregates, counts, comparisons, cohort summaries.
- Use type="predictive_modeling" when the user asks for:
    risk factors, top predictors, feature importance, mortality drivers,
    "what predicts X", "which features matter", correlation analysis,
    "compare outcomes", "identify top factors".
- For predictive_modeling steps:
    * The SQL MUST select ALL feature columns AND the target column.
    * Do NOT add LIMIT to the SQL — we need every row.
    * target_column must be the column name to predict (e.g. "hospital_expire_flag").
    * Choose features available in the schema: age_at_admit, icu_stay_count,
      icu_los_days, and any other relevant columns present in the schema.
- Limit to at most 4 steps total.
- If the question needs both a summary AND risk factors, emit one sql step + one predictive_modeling step.

{SCHEMA_DESCRIPTION}"""

ANALYTICS_REVIEW_SYSTEM = """You are the reviewer in a multi-agent healthcare analytics system.
Given the user question, plan, and executed step results, decide whether evidence is sufficient.

Return ONLY valid JSON:
{
  "status": "sufficient" | "partial" | "insufficient",
  "reasoning": "short explanation",
  "follow_up_steps": [
    {
      "name": "short_step_name",
      "type": "sql",
      "question": "one additional natural language SQL task",
      "must_succeed": false
    }
  ],
  "missing_requirements": []
}

Rules:
- Only propose follow-up steps if they are small and concrete.
- Propose at most 2 follow-up steps.
- Do not duplicate predictive_modeling steps already completed.
"""

ANALYTICS_FINDINGS_SYSTEM = """You are the findings agent in a multi-agent healthcare analytics system.
Produce grounded findings from the executed step results.

Return ONLY valid JSON:
{
  "answerability": "direct" | "partial" | "insufficient",
  "key_findings": ["grounded finding 1", "grounded finding 2"],
  "limitations": ["limitation 1"]
}

Rules:
- For predictive_modeling steps: report the top risk factors and model accuracy.
- For sql steps: report key aggregated statistics.
- Findings must only use executed results.
- Do not invent thresholds, causality, or future predictions beyond what the model returned.
"""

COMPLEX_HINTS = (
    "why", "driver", "drivers", "best match", "profile", "profiles",
    "compare", "versus", "vs", "across", "cohort", "outcome", "outcomes",
    "risk", "match", "relationship", "association", "combine", "both",
    "predict", "top factor", "top risk", "mortality", "feature importance",
    "which feature", "what predicts",
)


def _looks_complex(query: str) -> bool:
    lower = (query or "").lower()
    return any(keyword in lower for keyword in COMPLEX_HINTS)


def _normalise_plan(raw_plan: dict[str, Any], query: str) -> dict[str, Any]:
    steps = raw_plan.get("steps") or [{"name": "main", "type": "sql", "question": query, "must_succeed": True}]
    normalised_steps = []
    for index, step in enumerate(steps[:4], 1):
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type") or "sql").lower()
        normalised_step: dict[str, Any] = {
            "name": str(step.get("name") or f"step_{index}").strip() or f"step_{index}",
            "type": step_type,
            "must_succeed": bool(step.get("must_succeed", True)),
        }
        if step_type == "predictive_modeling":
            normalised_step["sql"] = str(step.get("sql") or "").strip()
            normalised_step["target_column"] = str(step.get("target_column") or "").strip()
            normalised_step["question"] = str(step.get("question") or "").strip()
        else:
            normalised_step["type"] = "sql"
            normalised_step["question"] = str(step.get("question") or query).strip()
            normalised_step["sql"] = ""
            normalised_step["target_column"] = ""

        normalised_steps.append(normalised_step)

    return {
        "execution_mode": str(raw_plan.get("execution_mode") or ("multi_step" if _looks_complex(query) else "simple")),
        "answerability": str(raw_plan.get("answerability") or "direct"),
        "reasoning": str(raw_plan.get("reasoning") or ""),
        "steps": normalised_steps or [{"name": "main", "type": "sql", "question": query, "must_succeed": True}],
        "missing_requirements": [str(item) for item in raw_plan.get("missing_requirements", [])][:6],
    }


def _serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import decimal
    serialised: list[dict[str, Any]] = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                converted[key] = value.isoformat()
            elif isinstance(value, decimal.Decimal):
                converted[key] = float(value)
            else:
                converted[key] = value
        serialised.append(converted)
    return serialised


def _build_step_summary(
    step_name: str, rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return {}, [], {}

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    summary_stats: dict[str, dict[str, float]] = {}
    outliers: list[dict[str, Any]] = []
    chart_data: dict[str, Any] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        mean = float(series.mean())
        std = float(series.std()) if len(series) > 1 else 0.0
        summary_stats[f"{step_name}.{col}"] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "count": int(series.count()),
        }
        if std > 0:
            mask = (series - mean).abs() > 2 * std
            for idx in series[mask].index[:5]:
                row_data = df.loc[idx].to_dict()
                for key, value in row_data.items():
                    if hasattr(value, "item"):
                        row_data[key] = value.item()
                outliers.append({"step": step_name, "row": row_data, "column": col, "value": float(series[idx])})

    if numeric_cols and non_numeric_cols:
        x_col = non_numeric_cols[0]
        y_col = numeric_cols[0]
        chart_data = {
            "step": step_name,
            "x": df[x_col].astype(str).tolist()[:100],
            "y": df[y_col].fillna(0).tolist()[:100],
            "x_label": x_col,
            "y_label": y_col,
        }

    return summary_stats, outliers, chart_data


def _step_context_for_llm(step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for step in step_results:
        step_type = step.get("type", "sql")
        if step_type == "predictive_modeling":
            # Pass a compact predictive summary — no raw rows
            compact.append({
                "name": step["name"],
                "type": "predictive_modeling",
                "target_column": step.get("target_column"),
                "model_type": step.get("model_type"),
                "metric_name": step.get("metric_name"),
                "metric_value": step.get("metric_value"),
                "row_count": step.get("row_count", 0),
                "feature_importances": step.get("feature_importances", [])[:5],
                "error": step.get("error"),
            })
        else:
            rows_preview = []
            for row in (step.get("rows_preview") or [])[:3]:
                compact_row = {}
                for key, value in row.items():
                    if isinstance(value, str) and len(value) > 120:
                        compact_row[key] = value[:120] + "..."
                    else:
                        compact_row[key] = value
                rows_preview.append(compact_row)
            compact.append({
                "name": step["name"],
                "type": "sql",
                "question": step.get("question", ""),
                "must_succeed": step.get("must_succeed", True),
                "row_count": step.get("row_count", 0),
                "error": step.get("error"),
                "sql": (step.get("sql") or "")[:600],
                "rows_preview": rows_preview,
            })
    return compact


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

async def _plan_steps(query: str) -> dict[str, Any]:
    """Ask the LLM to produce a structured execution plan (sql + predictive steps)."""
    if not _looks_complex(query):
        return {
            "execution_mode": "simple",
            "answerability": "direct",
            "reasoning": "Straightforward aggregate/grouped analytics query.",
            "steps": [{"name": "main", "type": "sql", "question": query, "must_succeed": True, "sql": "", "target_column": ""}],
            "missing_requirements": [],
        }

    try:
        plan = await chat_json(
            model=MAIN_MODEL,
            ollama_model=OLLAMA_MAIN_MODEL,
            system_prompt=ANALYTICS_PLAN_SYSTEM,
            user_prompt=query,
            max_tokens=1500,
            temperature=0.0,
        )
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return a JSON object")
        return _normalise_plan(plan, query)
    except Exception as exc:
        logger.warning("Analytics planner failed (%s); falling back to direct execution", exc)
        return {
            "execution_mode": "simple",
            "answerability": "direct",
            "reasoning": "Planner unavailable; falling back to single direct step.",
            "steps": [{"name": "main", "type": "sql", "question": query, "must_succeed": True, "sql": "", "target_column": ""}],
            "missing_requirements": [],
        }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _execute_steps(
    steps: list[dict[str, Any]],
    conn: asyncpg.Connection,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], int]:
    step_results: list[dict[str, Any]] = []
    combined_summary: dict[str, dict[str, float]] = {}
    combined_outliers: list[dict[str, Any]] = []
    first_chart: dict[str, Any] = {}
    first_data_table: list[dict[str, Any]] = []
    successful_steps = 0

    for step_idx, step in enumerate(steps):
        # Throttle: pause between steps so Groq TPM doesn't spike
        if step_idx > 0:
            await asyncio.sleep(5)

        step_type = step.get("type", "sql")

        if step_type == "predictive_modeling":
            result = await run_predictive_step(conn, step)
            step_results.append(result)
            if not result.get("error"):
                successful_steps += 1
                # Build a mini chart from feature importances for the frontend
                fi = result.get("feature_importances", [])
                if fi and not first_chart:
                    first_chart = {
                        "step": result["name"],
                        "x": [r["feature"] for r in fi],
                        "y": [r["importance"] for r in fi],
                        "x_label": "Feature",
                        "y_label": "Importance",
                        "chart_title": f"Top Risk Factors → {result.get('target_column', '')}",
                    }
                if fi and not first_data_table:
                    first_data_table = fi
            elif step.get("must_succeed"):
                logger.warning("Required predictive step failed: %s — %s", step["name"], result.get("error"))
        else:
            # Standard SQL step
            result = await generate_and_run_sql(step["question"], conn)
            rows = _serialize_rows(result.get("results", []))
            step_summary, step_outliers, step_chart = _build_step_summary(step["name"], rows)

            if rows and not first_data_table:
                first_data_table = rows[:100]
            if step_chart and not first_chart:
                first_chart = step_chart

            step_results.append({
                "name": step["name"],
                "type": "sql",
                "question": step.get("question", ""),
                "must_succeed": step.get("must_succeed", True),
                "sql": result.get("sql"),
                "row_count": result.get("row_count", 0),
                "error": result.get("error"),
                "rows_preview": rows[:10],
            })

            if rows and not result.get("error"):
                successful_steps += 1
                combined_summary.update(step_summary)
                combined_outliers.extend(step_outliers)
            elif step.get("must_succeed"):
                logger.warning("Required SQL step failed: %s", step["name"])

    return step_results, combined_summary, combined_outliers, first_chart, first_data_table, successful_steps


# ---------------------------------------------------------------------------
# Review & Findings
# ---------------------------------------------------------------------------

async def _review_execution(
    query: str, plan: dict[str, Any], step_results: list[dict[str, Any]]
) -> dict[str, Any]:
    if plan["execution_mode"] == "simple":
        status = "sufficient" if any((not step.get("error")) and step.get("row_count", 0) > 0 for step in step_results) else "insufficient"
        return {
            "status": status,
            "reasoning": "Single-step analytics path used.",
            "follow_up_steps": [],
            "missing_requirements": plan.get("missing_requirements", []),
        }

    try:
        review = await chat_json(
            model=MAIN_MODEL,
            ollama_model=OLLAMA_MAIN_MODEL,
            system_prompt=ANALYTICS_REVIEW_SYSTEM,
            user_prompt=(
                f"User question:\n{query}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Executed step results:\n{_step_context_for_llm(step_results)}"
            ),
            max_tokens=1500,
            temperature=0.0,
        )
        if not isinstance(review, dict):
            raise ValueError("Reviewer did not return a JSON object")
    except Exception as exc:
        logger.warning("Analytics reviewer failed (%s); using default review", exc)
        return {
            "status": "partial" if plan.get("answerability") == "partial" else "sufficient",
            "reasoning": "Reviewer unavailable; using executed step coverage.",
            "follow_up_steps": [],
            "missing_requirements": plan.get("missing_requirements", []),
        }

    follow_up_steps = []
    for index, step in enumerate((review.get("follow_up_steps") or [])[:2], 1):
        if isinstance(step, dict):
            follow_up_steps.append({
                "name": str(step.get("name") or f"follow_up_{index}").strip() or f"follow_up_{index}",
                "type": str(step.get("type") or "sql").lower(),
                "question": str(step.get("question") or "").strip(),
                "sql": str(step.get("sql") or "").strip(),
                "target_column": str(step.get("target_column") or "").strip(),
                "must_succeed": bool(step.get("must_succeed", False)),
            })

    return {
        "status": str(review.get("status") or "partial"),
        "reasoning": str(review.get("reasoning") or ""),
        "follow_up_steps": [s for s in follow_up_steps if s.get("question") or s.get("sql")],
        "missing_requirements": [str(item) for item in review.get("missing_requirements", [])][:6],
    }


async def _derive_findings(
    query: str,
    plan: dict[str, Any],
    review: dict[str, Any],
    step_results: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        findings = await chat_json(
            model=MAIN_MODEL,
            ollama_model=OLLAMA_MAIN_MODEL,
            system_prompt=ANALYTICS_FINDINGS_SYSTEM,
            user_prompt=(
                f"User question:\n{query}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Review verdict:\n{review}\n\n"
                f"Executed step results:\n{_step_context_for_llm(step_results)}"
            ),
            max_tokens=1500,
            temperature=0.0,
        )
        if not isinstance(findings, dict):
            raise ValueError("Findings agent did not return a JSON object")
        return {
            "answerability": str(findings.get("answerability") or plan.get("answerability") or "partial"),
            "key_findings": [str(item) for item in findings.get("key_findings", [])][:8],
            "limitations": [str(item) for item in findings.get("limitations", [])][:8],
        }
    except Exception as exc:
        logger.warning("Analytics findings agent failed (%s); using fallback findings", exc)
        return {
            "answerability": plan.get("answerability") or "partial",
            "key_findings": [],
            "limitations": [review.get("reasoning") or plan.get("reasoning") or "Limited analytics evidence."],
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def analyze(query: str, conn: asyncpg.Connection) -> dict:
    plan = await _plan_steps(query)

    # Throttle: brief pause between major pipeline stages to avoid Groq TPM spikes
    await asyncio.sleep(8)

    step_results, combined_summary, combined_outliers, first_chart, first_data_table, successful_steps = (
        await _execute_steps(plan["steps"], conn)
    )

    await asyncio.sleep(8)
    review = await _review_execution(query, plan, step_results)

    # Agentic loop: run follow-ups if evidence is still partial
    if review.get("follow_up_steps") and review.get("status") == "partial":
        await asyncio.sleep(8)
        follow_results, extra_summary, extra_outliers, extra_chart, extra_table, extra_ok = await _execute_steps(
            review["follow_up_steps"], conn
        )
        step_results.extend(follow_results)
        combined_summary.update(extra_summary)
        combined_outliers.extend(extra_outliers)
        if extra_chart and not first_chart:
            first_chart = extra_chart
        if extra_table and not first_data_table:
            first_data_table = extra_table
        successful_steps += extra_ok
        await asyncio.sleep(8)
        review = await _review_execution(query, plan, step_results)

    await asyncio.sleep(8)
    findings = await _derive_findings(query, plan, review, step_results)

    base = {
        "plan_reasoning": plan.get("reasoning", ""),
        "plan": plan,
        "review": review,
        "step_results": step_results,
        "key_findings": findings.get("key_findings", []),
        "limitations": findings.get("limitations", []),
        "missing_requirements": list(dict.fromkeys(
            plan.get("missing_requirements", []) + review.get("missing_requirements", [])
        )),
        "partial": findings.get("answerability") == "partial" or plan.get("answerability") == "partial",
        "sql": "\n\n".join(
            f"-- {s['name']}\n{s.get('sql', '')}" for s in step_results if s.get("sql")
        ),
    }

    if successful_steps == 0:
        return {
            **base,
            "summary_stats": {},
            "outliers": [],
            "data_table": [],
            "chart_data": {},
            "error": "No analytics steps produced usable data",
            "insufficiency_reason": review.get("reasoning") or plan.get("reasoning") or "No steps produced data.",
        }

    return {
        **base,
        "summary_stats": combined_summary,
        "outliers": combined_outliers[:10],
        "data_table": first_data_table,
        "chart_data": first_chart,
    }
