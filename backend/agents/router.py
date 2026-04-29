import json
import logging
from backend.config import ROUTER_MODEL, OLLAMA_ROUTER_MODEL
from backend.llm import chat

logger = logging.getLogger(__name__)

DOC_MARKERS = (
    "guideline",
    "nice",
    "clinical note",
    "clinical notes",
    "document",
    "documents",
    "pdf",
    "pathway",
    "recommend",
    "recommendation",
)

ANALYTICS_MARKERS = (
    "risk",
    "risk score",
    "threshold",
    "would have flagged",
    "before an acute event",
    "best match",
    "profile",
    "profiles",
    "cohort",
    "compare",
    "versus",
    " vs ",
    "outcome",
    "outcomes",
    "driver",
    "drivers",
    "trend",
    "outlier",
    "association",
)

SQL_MARKERS = (
    "how many",
    "count",
    "list",
    "show",
    "top ",
    "average",
    "mean",
    "sum",
    "group by",
)

SYSTEM_PROMPT = """You are a healthcare query classifier. Classify the user query into exactly one type. Return ONLY valid JSON, no other text.

Types:
  sql: a single-step count, filter, list, or simple aggregation from structured patient data (one SQL query suffices)
  rag: retrieving information from clinical notes, NICE guidelines, or PDF documents
  hybrid: requires BOTH structured patient data AND guideline/document retrieval together
  analytics: multi-step analysis — cohort comparisons, trend analysis, outcome drivers, outlier detection, cross-dataset questions (Synthea vs MIMIC), or EHG uterine signal analysis. Use analytics — not sql — when the question asks "why", "what drives", "compare across", "risk factors", "profile", or involves two datasets.

Disambiguation: use sql for "how many", "list", "show me" questions answered by a single aggregate. Use analytics for "compare", "what are the drivers", "across sites", "versus", or any question needing multiple SQL steps.

Available datasets: Synthea synthetic EHR (patients, encounters, conditions, medications, observations), MIMIC-III ICU (admissions, diagnoses, lab events, prescriptions), EHG uterine electrophysiology records (ehg_records, ehg_features).

Return: {"type": string, "reasoning": string, "requires_patient_data": bool, "requires_documents": bool, "time_sensitive": bool}"""


def _heuristic_classification(query: str) -> dict | None:
    lower = (query or "").lower()
    if not lower.strip():
        return None

    mentions_docs = any(marker in lower for marker in DOC_MARKERS)
    mentions_analytics = any(marker in lower for marker in ANALYTICS_MARKERS)
    mentions_sql = any(marker in lower for marker in SQL_MARKERS)
    cross_dataset = ("synthea" in lower and "mimic" in lower) or ("mimic" in lower and "ehg" in lower)

    if mentions_docs and (mentions_analytics or mentions_sql):
        return {
            "type": "hybrid",
            "reasoning": "Heuristic routing: the question mixes structured analysis with document retrieval.",
            "requires_patient_data": True,
            "requires_documents": True,
            "time_sensitive": False,
        }

    if mentions_docs:
        return {
            "type": "rag",
            "reasoning": "Heuristic routing: the question primarily asks for guidance or document-backed information.",
            "requires_patient_data": False,
            "requires_documents": True,
            "time_sensitive": False,
        }

    if cross_dataset or mentions_analytics:
        return {
            "type": "analytics",
            "reasoning": "Heuristic routing: the question needs multi-step cohort/outcome analytics rather than a single SQL aggregate.",
            "requires_patient_data": True,
            "requires_documents": False,
            "time_sensitive": False,
        }

    return None


async def classify_query(query: str) -> dict:
    if heuristic := _heuristic_classification(query):
        return heuristic

    try:
        raw = await chat(
            model=ROUTER_MODEL,
            ollama_model=OLLAMA_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Best-effort recovery: extract the first JSON object found in the output.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise
    except json.JSONDecodeError as e:
        logger.warning("Router JSON parse failed (%s); defaulting to hybrid", e)
        return {
            "type": "hybrid",
            "reasoning": "Could not parse router response; defaulting to hybrid",
            "requires_patient_data": True,
            "requires_documents": True,
            "time_sensitive": False,
        }
    except Exception as e:
        logger.error("Router error: %s", e)
        raise
