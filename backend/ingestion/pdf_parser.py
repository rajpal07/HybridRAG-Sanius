import asyncio
import json
import os
import logging
from typing import Optional

import asyncpg

from backend.config import RAW_DIR, LLAMA_PARSE_API_KEY, ROUTER_MODEL, OLLAMA_ROUTER_MODEL
from backend.database import get_pool
from backend.database import pgvector_available
from backend.ingestion.embedder import embed_text
from backend.llm import chat

logger = logging.getLogger(__name__)

PDF_DIR = os.path.join(RAW_DIR, "pdfs")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

METADATA_SYSTEM = """You are a clinical document analyser. Given a healthcare document excerpt, extract metadata.
Return ONLY valid JSON with these exact keys:
{
  "clinical_domains": ["up to 6 medical domains, e.g. CKD, diabetes, cardiovascular, prescribing, respiratory, pain, haematology, oncology"],
  "conditions": ["up to 6 specific conditions mentioned"],
  "drugs": ["up to 6 drug names or classes mentioned"],
  "summary": "one sentence describing what this document covers"
}
Return ONLY the JSON object, no other text."""


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return [c for c in chunks if c.strip()]


def _parse_with_pymupdf(path: str) -> str:
    try:
        import fitz
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        logger.error("PyMuPDF failed for %s: %s", path, e)
        return ""


def _parse_with_llamaparse(path: str) -> Optional[str]:
    if not LLAMA_PARSE_API_KEY:
        return None
    try:
        from llama_parse import LlamaParse
        parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY, result_type="text")
        documents = parser.load_data(path)
        return "\n\n".join(doc.text for doc in documents)
    except Exception as e:
        logger.warning("LlamaParse failed for %s (%s); falling back to PyMuPDF", path, e)
        return None


def _classify_doc_type(filename: str) -> str:
    name = filename.lower()
    if "guideline" in name or "nice" in name or "cg" in name or "qs" in name:
        return "guideline"
    if "pathway" in name:
        return "pathway"
    if "paper" in name or "journal" in name:
        return "paper"
    return "guideline"


async def _extract_doc_metadata(text: str) -> dict:
    excerpt = " ".join(text.split()[:1500])
    try:
        raw = await chat(
            model=ROUTER_MODEL,
            ollama_model=OLLAMA_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": METADATA_SYSTEM},
                {"role": "user", "content": excerpt},
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.warning("Metadata extraction failed (%s); proceeding without metadata", e)
        return {}


async def ingest_pdf(conn: asyncpg.Connection, path: str):
    filename = os.path.basename(path)
    logger.info("Parsing PDF: %s", filename)

    use_pgvector = await pgvector_available(conn)

    # Upsert behavior: replace existing chunks for this source file.
    try:
        await conn.execute("DELETE FROM documents WHERE source_name = $1", filename)
    except Exception:
        # Backward compatibility: older schemas may not have source_name column populated consistently.
        pass

    text = await asyncio.to_thread(_parse_with_llamaparse, path)
    if not text:
        text = await asyncio.to_thread(_parse_with_pymupdf, path)

    if not text.strip():
        logger.warning("No text extracted from %s", filename)
        return

    chunks = _chunk_text(text)
    doc_type = _classify_doc_type(filename)
    doc_metadata = await _extract_doc_metadata(text)
    metadata_json = json.dumps(doc_metadata)
    logger.info("  %d chunks from %s | domains: %s", len(chunks), filename,
                doc_metadata.get("clinical_domains", []))

    for idx, chunk in enumerate(chunks):
        try:
            embedding = await asyncio.to_thread(embed_text, chunk)
            if use_pgvector:
                await conn.execute(
                    """INSERT INTO documents (source_name, doc_type, chunk_index, content, embedding, metadata)
                       VALUES ($1,$2,$3,$4,$5::vector,$6)""",
                    filename,
                    doc_type,
                    idx,
                    chunk,
                    str(embedding),
                    metadata_json,
                )
            else:
                await conn.execute(
                    """INSERT INTO documents (source_name, doc_type, chunk_index, content, embedding, metadata)
                       VALUES ($1,$2,$3,$4,$5,$6)""",
                    filename,
                    doc_type,
                    idx,
                    chunk,
                    embedding,
                    metadata_json,
                )
        except Exception as e:
            logger.error("Failed to insert chunk %d of %s: %s", idx, filename, e)

    logger.info("  Ingested %s (%d chunks)", filename, len(chunks))


async def ingest_all_pdfs():
    if not os.path.exists(PDF_DIR):
        logger.warning("PDF directory not found at %s", PDF_DIR)
        return

    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not files:
        logger.warning("No PDF files found in %s", PDF_DIR)
        return

    pool = await get_pool()
    async with pool.acquire() as conn:
        for fname in files:
            await ingest_pdf(conn, os.path.join(PDF_DIR, fname))

    logger.info("All PDFs ingested.")
