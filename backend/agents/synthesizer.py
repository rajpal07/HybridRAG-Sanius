import re
import logging
from typing import Optional
from backend.config import MAIN_MODEL, OLLAMA_MAIN_MODEL
from backend.llm import chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical AI assistant for Sanius Health Intelligence.
Your role is to synthesise information from structured patient data, clinical notes, and NHS guidelines to answer clinician queries.

RULES (non-negotiable):
1. Generate answers ONLY from the provided context. Every factual claim must be followed by [Source: X].
2. If the provided context is insufficient, respond exactly: "Insufficient data in the current dataset to answer this question reliably."
3. Never speculate. Never hallucinate clinical information not present in the context.
4. Use plain clinical English. Be concise and precise.
5. When citing SQL data, use [Source: structured_data]. When citing analytics results, use [Source: analytics]. When citing chunks, use [Source: <source_name>].
6. End with a confidence assessment: HIGH (all claims directly supported), MEDIUM (partial support), or LOW (limited data)."""


def _format_sql_context(sql_result: Optional[dict]) -> str:
    if not sql_result or sql_result.get("error"):
        return ""
    if not sql_result.get("results"):
        return f"SQL Query Results (0 rows total):\n  No matching records found."
    rows = sql_result["results"][:20]
    lines = [f"SQL Query Results ({sql_result['row_count']} rows total):"]
    for row in rows:
        lines.append("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(lines)


def _format_rag_context(rag_result: Optional[dict]) -> str:
    if not rag_result or not rag_result.get("chunks"):
        return ""
    lines = ["Retrieved Document Chunks:"]
    for i, chunk in enumerate(rag_result["chunks"], 1):
        source = chunk.get("source", "unknown")
        doc_type = chunk.get("doc_type", "")
        sim = chunk.get("similarity", 0)
        content = chunk.get("content", "")[:500]
        lines.append(f"\n[Chunk {i} | Source: {source} | Type: {doc_type} | Similarity: {sim:.3f}]")
        lines.append(content)
    return "\n".join(lines)


def _format_analytics_context(analytics_result: Optional[dict]) -> str:
    if not analytics_result:
        return ""
    lines = ["Analytics Results:"]
    if analytics_result.get("plan_reasoning"):
        lines.append(f"Plan reasoning: {analytics_result['plan_reasoning']}")
    if analytics_result.get("review", {}).get("reasoning"):
        lines.append(f"Review reasoning: {analytics_result['review']['reasoning']}")
    if analytics_result.get("missing_requirements"):
        lines.append("Missing requirements: " + ", ".join(analytics_result["missing_requirements"]))
    if analytics_result.get("key_findings"):
        lines.append("Key findings:")
        for finding in analytics_result["key_findings"][:6]:
            lines.append(f"  - {finding}")
    if analytics_result.get("limitations"):
        lines.append("Limitations:")
        for limitation in analytics_result["limitations"][:6]:
            lines.append(f"  - {limitation}")
    step_results = analytics_result.get("step_results", [])
    if step_results:
        lines.append("Executed analytics steps:")
        for step in step_results[:4]:
            lines.append(
                f"  - {step.get('name')}: rows={step.get('row_count', 0)}, error={step.get('error') or 'none'}"
            )
            if step.get("rows_preview"):
                preview = step["rows_preview"][:3]
                for row in preview:
                    lines.append("    " + ", ".join(f"{k}={v}" for k, v in row.items()))
    stats = analytics_result.get("summary_stats", {})
    for col, s in stats.items():
        if not isinstance(s, dict):
            continue
        lines.append(f"  {col}: mean={s.get('mean','?')}, std={s.get('std','?')}, min={s.get('min','?')}, max={s.get('max','?')}, n={s.get('count','?')}")
    outliers = analytics_result.get("outliers", [])
    if outliers:
        lines.append(f"Outliers detected: {len(outliers)} rows beyond 2 standard deviations")
    return "\n".join(lines)


def _extract_citations(text: str) -> list[str]:
    return re.findall(r"\[Source:\s*([^\]]+)\]", text)


def _strip_embedded_confidence(text: str) -> str:
    return re.sub(
        r"\n+\s*Confidence(?: assessment)?:\s*(HIGH|MEDIUM|LOW)\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


def _assess_confidence(text: str, sql_result, rag_result, analytics_result) -> str:
    if "Insufficient data" in text:
        return "low"
    has_sql = sql_result and sql_result.get("error") is None
    has_rag = rag_result and len(rag_result.get("chunks", [])) > 0
    has_analytics = analytics_result and (
        analytics_result.get("summary_stats") or 
        analytics_result.get("key_findings") or
        analytics_result.get("step_results")
    )
    
    sources_used = sum([bool(has_sql), bool(has_rag), bool(has_analytics)])
    if sources_used >= 2:
        return "high"
    if sources_used == 1:
        return "medium"
    return "low"


def _friendly_key(key: str) -> str:
    return key.replace("_", " ")


def _fallback_answer(router_result: dict, sql_result, rag_result, analytics_result) -> dict | None:
    if analytics_result and analytics_result.get("summary_stats"):
        findings = analytics_result.get("key_findings") or []
        limitations = analytics_result.get("limitations") or []
        lines = []
        if findings:
            lines.extend(f"{finding} [Source: analytics]." for finding in findings[:3])
        if limitations:
            lines.append("Limitations: " + "; ".join(limitations[:2]) + " [Source: analytics].")
        if lines:
            answer = " ".join(lines)
            return {
                "answer": answer,
                "citations": _extract_citations(answer),
                "confidence": "medium" if findings else "low",
                "sources_used": ["analytics"],
                "sql_used": None,
            }

    if sql_result and sql_result.get("results"):
        first_row = sql_result["results"][0]
        facts = [f"{_friendly_key(key)}: {value}" for key, value in first_row.items()]
        answer = f"Structured data result - {'; '.join(facts)} [Source: structured_data]."
        if router_result.get("requires_documents") and not (rag_result and rag_result.get("chunks")):
            answer += " No relevant document guidance was retrieved from the current corpus."
        return {
            "answer": answer,
            "citations": _extract_citations(answer),
            "confidence": "medium" if not router_result.get("requires_documents") else "low",
            "sources_used": ["structured_data"],
            "sql_used": sql_result.get("sql"),
        }

    return None


async def synthesize(
    query: str,
    router_result: dict,
    sql_result: Optional[dict] = None,
    rag_result: Optional[dict] = None,
    analytics_result: Optional[dict] = None,
) -> dict:
    context_parts = []
    if sql_ctx := _format_sql_context(sql_result):
        context_parts.append(sql_ctx)
    if rag_ctx := _format_rag_context(rag_result):
        context_parts.append(rag_ctx)
    if analytics_ctx := _format_analytics_context(analytics_result):
        context_parts.append(analytics_ctx)

    if not context_parts:
        return {
            "answer": "Insufficient data in the current dataset to answer this question reliably.",
            "citations": [],
            "confidence": "low",
            "sources_used": [],
            "sql_used": sql_result.get("sql") if sql_result else None,
        }

    user_message = f"""Clinical query: {query}

Query classification: {router_result.get('type', 'hybrid')} — {router_result.get('reasoning', '')}

Context:
{chr(10).join(context_parts)}

Please answer the clinical query using only the context above. Cite every fact."""

    try:
        answer = await chat(
            model=MAIN_MODEL,
            ollama_model=OLLAMA_MAIN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.1,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Synthesis failed: %s", e)
        fallback = _fallback_answer(router_result, sql_result, rag_result, analytics_result)
        if fallback:
            return fallback
        return {
            "answer": "Insufficient data in the current dataset to answer this question reliably.",
            "citations": [],
            "confidence": "low",
            "sources_used": [],
            "sql_used": sql_result.get("sql") if sql_result else None,
        }

    cleaned_answer = _strip_embedded_confidence(answer)
    citations = _extract_citations(cleaned_answer)
    sources_used = []
    if sql_result and sql_result.get("error") is None:
        sources_used.append("structured_data")
    if rag_result and rag_result.get("chunks"):
        sources_used.extend(c["source"] for c in rag_result["chunks"])
    if analytics_result and not analytics_result.get("error"):
        sources_used.append("analytics")

    return {
        "answer": cleaned_answer,
        "citations": citations,
        "confidence": _assess_confidence(cleaned_answer, sql_result, rag_result, analytics_result),
        "sources_used": list(set(sources_used)),
        "sql_used": sql_result.get("sql") if sql_result else None,
    }
