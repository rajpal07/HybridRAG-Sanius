import asyncio
import json
import logging
import re
import asyncpg
from backend.config import ROUTER_MODEL, OLLAMA_ROUTER_MODEL
from backend.database import pgvector_available
from backend.llm import chat

DOMAIN_MAP: dict[str, list[str]] = {
    "renal": ["kidney", "renal", "ckd", "egfr", "creatinine", "nephropathy", "dialysis", "aki", "glomerular", "renal insufficiency"],
    "diabetes": ["diabetes", "diabetic", "hba1c", "glucose", "metformin", "insulin", "glycaemic", "glycemic"],
    "cardiovascular": ["heart failure", "cardiac", "coronary", "atrial fibrillation", "hypertension",
                       "blood pressure", "statin", "ace inhibitor", "lisinopril", "ramipril", "losartan",
                       "arb", "beta blocker", "anticoagulant", "warfarin", "apixaban"],
    "prescribing": ["prescri", "polypharmacy", "drug interaction", "nsaid", "ibuprofen", "naproxen",
                    "analgesic", "formulary", "dosing", "contraindication", "medication safety"],
    "respiratory": ["copd", "asthma", "respiratory", "inhaler", "bronchodilator", "spirometry"],
    "pain": ["pain management", "analgesic", "opioid", "palliative", "chronic pain", "neuropathic"],
    "mental_health": ["depression", "anxiety", "mental health", "psychiatric", "dementia", "cognitive"],
    "obstetrics": ["pregnan", "maternal", "preterm", "gestation", "ehg", "labour", "antenatal"],
    "oncology": ["cancer", "tumour", "tumor", "oncolog", "chemotherapy", "malignant"],
    "haematology": ["anaemia", "anemia", "sickle cell", "haematolog", "blood disorder", "clotting", "thrombosis"],
}

QUERY_EXPANSIONS: dict[str, list[str]] = {
    "sickle cell": [
        "sickle cell",
        "acute painful episode",
        "pain crisis",
        "vaso occlusive",
        "cg143",
        "qs58",
    ],
    "pain management": [
        "pain management",
        "analgesia",
        "analgesic",
        "opioid",
        "nsaid",
        "paracetamol",
        "assessment",
    ],
    "nice guideline": [
        "nice guideline",
        "nice",
        "guideline",
        "quality standard",
    ],
}

GENERIC_QUERY_TOKENS = {
    "what", "does", "guidance", "guideline", "nice", "patient", "patients", "clinical",
    "current", "dataset", "combination", "condition", "conditions", "code", "codes",
    "question", "elderly", "older", "adult", "adults", "data", "with", "from", "into",
    "about", "those", "these", "also", "many", "over", "under", "than", "have", "that",
    "this", "which", "their", "risk", "synthea", "mimic", "prescribed", "concurrently",
}


def _extract_query_domains(query: str) -> list[str]:
    q = query.lower()
    return [domain for domain, keywords in DOMAIN_MAP.items() if any(kw in q for kw in keywords)]


def _query_variants(query: str) -> list[str]:
    variants: list[str] = []
    base = (query or "").strip()
    if base:
        variants.append(base)

    lower = base.lower()
    expansions: list[str] = []
    for trigger, extra_terms in QUERY_EXPANSIONS.items():
        if trigger in lower:
            expansions.extend(extra_terms)

    if "sickle" in lower and "cg143" not in lower:
        expansions.extend(["sickle cell", "acute painful episode", "cg143"])
    if "guideline" in lower and "nice" not in lower:
        expansions.extend(["nice guideline"])
    if "pain" in lower and "analges" not in lower:
        expansions.extend(["analgesia", "pain management"])

    if expansions:
        expanded = " ".join(dict.fromkeys([base, *expansions]))
        variants.append(expanded)

    return list(dict.fromkeys(v for v in variants if v.strip()))


def _source_name_hints(query: str) -> list[str]:
    q = (query or "").lower()
    hints: list[str] = []
    if "sickle" in q:
        hints.extend(["%sickle%", "%cg143%", "%qs58%", "%west_london%"])
    if "nice" in q or "guideline" in q:
        hints.extend(["%nice%", "%guideline%", "%qs58%", "%cg143%"])
    if "west london" in q:
        hints.append("%west_london%")
    return list(dict.fromkeys(hints))


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9\-\+]+", (text or "").lower())
        if len(token) >= 3 and token not in GENERIC_QUERY_TOKENS
    }


def _query_mentions_docs(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in ("nice", "guideline", "guidance", "document", "documents", "pdf"))


def _document_metadata_overlap(candidate: dict, query: str) -> set[str]:
    metadata = candidate.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    metadata_parts = [candidate.get("source", "")]
    for key in ("conditions", "drugs"):
        values = metadata.get(key) or []
        if isinstance(values, list):
            metadata_parts.extend(str(value) for value in values)
    return _tokenize(query).intersection(_tokenize(" ".join(metadata_parts)))


def _filter_document_candidates(query: str, candidates: list[dict], domains: list[str]) -> list[dict]:
    if not candidates:
        return []
    if not _query_mentions_docs(query):
        return candidates

    filtered: list[dict] = []
    for candidate in candidates:
        if candidate.get("doc_type") == "clinical_note":
            filtered.append(candidate)
            continue

        overlap = _document_metadata_overlap(candidate, query)
        metadata = candidate.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        doc_domains = {str(item).lower() for item in (metadata.get("clinical_domains") or [])}
        domain_overlap = set(domains).intersection(doc_domains) if domains else set()

        if overlap:
            filtered.append(candidate)
            continue
        if domains and domain_overlap:
            filtered.append(candidate)

    return filtered

logger = logging.getLogger(__name__)

RERANK_SYSTEM = """You are a relevance ranker for a healthcare RAG system.
Given a query and a list of document chunks (each with an id), return the top 5 most relevant chunk IDs as a JSON array.
Return ONLY a valid JSON array of integers, e.g. [3, 1, 7, 2, 9]. No other text."""


def _fts_or_query(q: str) -> str:
    # Convert a natural language question into a forgiving OR-style search query.
    # Example: "NICE CG143 sickle cell pain crisis" -> "nice OR cg143 OR sickle OR cell OR pain OR crisis"
    tokens = []
    for raw in (q or "").lower().replace("'", " ").split():
        t = "".join(ch for ch in raw if ch.isalnum())
        if len(t) < 3:
            continue
        if t in ("what", "does", "from", "with", "into", "your", "about", "use", "only", "then", "cite"):
            continue
        tokens.append(t)
    # Keep the query bounded.
    tokens = tokens[:10]
    if not tokens:
        return (q or "").strip()
    return " OR ".join(tokens)


def _merge_candidates(existing: dict[tuple[str, int], dict], rows: list[dict]) -> None:
    for row in rows:
        key = (row["doc_type"], row["id"])
        current = existing.get(key)
        if current is None or row["similarity"] > current["similarity"]:
            existing[key] = row


def _prune_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top_similarity = candidates[0]["similarity"]
    if top_similarity <= 0:
        return candidates[: max(10, top_k * 4)]
    if top_similarity < 0.1:
        return candidates[: max(12, top_k * 5)]

    floor = max(0.02, top_similarity * 0.2)
    pruned = [candidate for candidate in candidates if candidate["similarity"] >= floor]
    return pruned[: max(12, top_k * 5)]


async def retrieve(query: str, conn: asyncpg.Connection, top_k: int = 5) -> dict:
    doc_guidance_query = _query_mentions_docs(query)
    if await pgvector_available(conn):
        from backend.ingestion.embedder import embed_text
        query_variants = _query_variants(query)
        domains = _extract_query_domains(query)
        all_kws = list({kw for d in domains for kw in DOMAIN_MAP.get(d, [])})
        pattern = "|".join(re.escape(kw) for kw in all_kws[:25]) if all_kws else ""
        source_hints = _source_name_hints(query)
        candidate_map: dict[tuple[str, int], dict] = {}
        query_embedding_dim = 0

        for query_variant in query_variants:
            query_embedding = await asyncio.to_thread(embed_text, query_variant)
            query_embedding_dim = max(query_embedding_dim, len(query_embedding))
            embedding_str = str(query_embedding)

            note_rows = []
            if not doc_guidance_query:
                note_rows = await conn.fetch(
                    """SELECT id, patient_id, note_type, redacted_content AS content,
                              'clinical_note' AS doc_type,
                              1 - (embedding <=> $1::vector) AS similarity
                       FROM clinical_notes
                       WHERE embedding IS NOT NULL
                       ORDER BY embedding <=> $1::vector
                       LIMIT 10""",
                    embedding_str,
                )

            doc_rows = []
            if domains:
                doc_rows = list(
                    await conn.fetch(
                        """SELECT id, NULL::INTEGER AS patient_id, doc_type, content,
                                  metadata,
                                  source_name AS doc_type_label,
                                  source_name,
                                  (
                                      1 - (embedding <=> $1::vector)
                                      + CASE WHEN array_length($3::text[], 1) IS NOT NULL
                                              AND source_name ILIKE ANY($3::text[])
                                             THEN 0.15 ELSE 0 END
                                  ) AS similarity
                           FROM documents
                           WHERE embedding IS NOT NULL
                             AND (
                                  metadata IS NULL
                                  OR metadata = '{}'::jsonb
                                  OR jsonb_exists_any(metadata->'clinical_domains', $2::text[])
                                  OR metadata::text ~* $4
                                  OR (array_length($3::text[], 1) IS NOT NULL AND source_name ILIKE ANY($3::text[]))
                             )
                           ORDER BY similarity DESC
                           LIMIT 10""",
                        embedding_str,
                        domains,
                        source_hints,
                        pattern,
                    )
                )
            if not doc_rows:
                doc_rows = await conn.fetch(
                    """SELECT id, NULL::INTEGER AS patient_id, doc_type, content,
                              metadata,
                              source_name AS doc_type_label,
                              source_name,
                              (
                                  1 - (embedding <=> $1::vector)
                                  + CASE WHEN array_length($2::text[], 1) IS NOT NULL
                                          AND source_name ILIKE ANY($2::text[])
                                         THEN 0.15 ELSE 0 END
                              ) AS similarity
                       FROM documents
                       WHERE embedding IS NOT NULL
                       ORDER BY similarity DESC
                       LIMIT 10""",
                    embedding_str,
                    source_hints,
                )

            _merge_candidates(
                candidate_map,
                [
                    {
                        "id": row["id"],
                        "content": row["content"] or "",
                        "source": f"clinical_note_{row['id']}",
                        "similarity": float(row["similarity"]),
                        "doc_type": "clinical_note",
                        "patient_id": row["patient_id"],
                    }
                    for row in note_rows
                ],
            )
            _merge_candidates(
                candidate_map,
                [
                    {
                        "id": row["id"],
                        "content": row["content"] or "",
                        "source": row["doc_type_label"] or f"document_{row['id']}",
                        "similarity": float(row["similarity"]),
                        "doc_type": row["doc_type"] or "document",
                        "patient_id": None,
                        "metadata": row["metadata"] or {},
                    }
                    for row in doc_rows
                ],
            )
    else:
        # Fallback: full-text search (no pgvector extension installed)
        domains = _extract_query_domains(query)
        all_kws = list({kw for d in domains for kw in DOMAIN_MAP.get(d, [])})
        pattern = "|".join(re.escape(kw) for kw in all_kws[:25]) if all_kws else ""
        source_hints = _source_name_hints(query)
        candidate_map: dict[tuple[str, int], dict] = {}
        query_embedding_dim = 0
        for query_variant in _query_variants(query):
            q = _fts_or_query(query_variant)
            note_rows = []
            if not doc_guidance_query:
                note_rows = await conn.fetch(
                    """SELECT id, patient_id, note_type, redacted_content AS content,
                              'clinical_note' AS doc_type,
                              ts_rank(
                                  to_tsvector('english', coalesce(redacted_content, '')),
                                  websearch_to_tsquery('english', $1)
                              ) AS similarity
                       FROM clinical_notes
                       WHERE to_tsvector('english', coalesce(redacted_content, '')) @@ websearch_to_tsquery('english', $1)
                       ORDER BY similarity DESC
                       LIMIT 10""",
                    q,
                )

            doc_rows = []
            if domains:
                doc_rows = list(
                    await conn.fetch(
                        """SELECT id, NULL::INTEGER AS patient_id, doc_type, content,
                                  source_name AS doc_type_label,
                                  metadata,
                                  (
                                      ts_rank(
                                          to_tsvector('english', coalesce(content, '')),
                                          websearch_to_tsquery('english', $1)
                                      )
                                      + CASE WHEN array_length($4::text[], 1) IS NOT NULL
                                              AND source_name ILIKE ANY($4::text[])
                                             THEN 0.15 ELSE 0 END
                                  ) AS similarity
                           FROM documents
                           WHERE (
                                  to_tsvector('english', coalesce(content, '')) @@ websearch_to_tsquery('english', $1)
                                  OR (array_length($4::text[], 1) IS NOT NULL AND source_name ILIKE ANY($4::text[]))
                           )
                             AND (
                                  metadata IS NULL
                                  OR metadata = '{}'::jsonb
                                  OR jsonb_exists_any(metadata->'clinical_domains', $2::text[])
                                  OR metadata::text ~* $3
                                  OR (array_length($4::text[], 1) IS NOT NULL AND source_name ILIKE ANY($4::text[]))
                             )
                           ORDER BY similarity DESC
                           LIMIT 10""",
                        q,
                        domains,
                        pattern,
                        source_hints,
                    )
                )

            if not doc_rows:
                doc_rows = await conn.fetch(
                    """SELECT id, NULL::INTEGER AS patient_id, doc_type, content,
                              source_name AS doc_type_label,
                              metadata,
                              (
                                  ts_rank(
                                      to_tsvector('english', coalesce(content, '')),
                                      websearch_to_tsquery('english', $1)
                                  )
                                  + CASE WHEN array_length($2::text[], 1) IS NOT NULL
                                          AND source_name ILIKE ANY($2::text[])
                                         THEN 0.15 ELSE 0 END
                              ) AS similarity
                       FROM documents
                       WHERE to_tsvector('english', coalesce(content, '')) @@ websearch_to_tsquery('english', $1)
                          OR (array_length($2::text[], 1) IS NOT NULL AND source_name ILIKE ANY($2::text[]))
                       ORDER BY similarity DESC
                       LIMIT 10""",
                    q,
                    source_hints,
                )

            _merge_candidates(
                candidate_map,
                [
                    {
                        "id": row["id"],
                        "content": row["content"] or "",
                        "source": f"clinical_note_{row['id']}",
                        "similarity": float(row["similarity"]),
                        "doc_type": "clinical_note",
                        "patient_id": row["patient_id"],
                    }
                    for row in note_rows
                ],
            )
            _merge_candidates(
                candidate_map,
                [
                    {
                        "id": row["id"],
                        "content": row["content"] or "",
                        "source": row["doc_type_label"] or f"document_{row['id']}",
                        "similarity": float(row["similarity"]),
                        "doc_type": row["doc_type"] or "document",
                        "patient_id": None,
                        "metadata": row["metadata"] or {},
                    }
                    for row in doc_rows
                ],
            )

    candidates = _filter_document_candidates(query, list(candidate_map.values()), domains if 'domains' in locals() else [])
    top_20 = _prune_candidates(candidates, top_k)

    logger.info(
        "RAG retrieve query=%r variants=%d domains=%s candidates=%d top=%s",
        query,
        len(_query_variants(query)),
        domains if 'domains' in locals() else [],
        len(candidates),
        [
            {
                "source": c["source"],
                "similarity": round(c["similarity"], 4),
            }
            for c in top_20[:5]
        ],
    )

    if not top_20:
        return {"chunks": [], "query_embedding_dim": query_embedding_dim}

    # Rerank with LLM
    top_ids = await _rerank(query, top_20, top_k)
    id_to_chunk = {c["id"]: c for c in top_20}
    reranked = [id_to_chunk[i] for i in top_ids if i in id_to_chunk]

    return {"chunks": reranked[:top_k], "query_embedding_dim": query_embedding_dim}


async def _rerank(query: str, candidates: list[dict], top_k: int) -> list[int]:
    try:
        candidate_text = "\n\n".join(
            f"[ID {c['id']}] ({c['doc_type']}, similarity={c['similarity']:.3f}):\n{c['content'][:300]}"
            for c in candidates
        )
        user_msg = f"Query: {query}\n\nCandidates:\n{candidate_text}"

        raw = await chat(
            model=ROUTER_MODEL,
            ollama_model=OLLAMA_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
        ids = json.loads(raw)
        return [int(i) for i in ids]
    except Exception as e:
        logger.warning("Reranking failed (%s); using similarity order", e)
        return [c["id"] for c in candidates[:top_k]]
