from __future__ import annotations

import asyncio
import importlib
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


REQUIRED_IMPORTS = [
    "fastapi",
    "uvicorn",
    "asyncpg",
    "psycopg2",
    "pgvector",
    "groq",
    "ollama",
    "presidio_analyzer",
    "presidio_anonymizer",
    "spacy",
    "pandas",
    "openpyxl",
    "llama_parse",
    "unstructured",
    "fitz",
    "streamlit",
    "httpx",
]

SPACY_MODELS = ["en_core_web_lg", "en_core_web_sm"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _mask(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return ("*" * (len(s) - keep)) + s[-keep:]


def _mask_db_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = parsed.port or ""
    db = (parsed.path or "").lstrip("/")
    return f"{parsed.scheme}://{user}:***@{host}:{port}/{db}"


def _tcp_probe(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_imports() -> CheckResult:
    failed: list[str] = []
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{mod}: {e}")
    if failed:
        return CheckResult("python_imports", False, "Missing/broken imports:\n  " + "\n  ".join(failed))
    return CheckResult("python_imports", True, "All required imports succeeded")


def _check_spacy_models() -> CheckResult:
    try:
        import spacy
    except Exception as e:  # noqa: BLE001
        return CheckResult("spacy_models", False, f"spaCy import failed: {e}")

    missing: list[str] = []
    for name in SPACY_MODELS:
        try:
            spacy.load(name)
        except Exception:
            missing.append(name)
    if missing:
        return CheckResult("spacy_models", False, "Missing models: " + ", ".join(missing))
    return CheckResult("spacy_models", True, "spaCy models OK: " + ", ".join(SPACY_MODELS))


async def _check_postgres(db_url: str) -> CheckResult:
    host = "localhost"
    port = 5432
    parsed = urlparse(db_url)
    if parsed.hostname:
        host = parsed.hostname
    if parsed.port:
        port = parsed.port

    if not _tcp_probe(host, port):
        return CheckResult("postgres", False, f"TCP connection failed to {host}:{port} (is Postgres running?)")

    try:
        import asyncpg
    except Exception as e:  # noqa: BLE001
        return CheckResult("postgres", False, f"asyncpg import failed: {e}")

    try:
        conn = await asyncpg.connect(db_url, timeout=3)
        try:
            await conn.fetchval("select 1")
        finally:
            await conn.close()
        return CheckResult("postgres", True, f"Connected OK ({_mask_db_url(db_url)})")
    except Exception as e:  # noqa: BLE001
        return CheckResult("postgres", False, f"Connect failed ({_mask_db_url(db_url)}): {e}")


def _check_ollama(ollama_base_url: str) -> CheckResult:
    # Default Ollama is localhost:11434
    parsed = urlparse(ollama_base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    if not _tcp_probe(host, port):
        return CheckResult("ollama", False, f"Not reachable at {host}:{port} (install/start Ollama)")
    return CheckResult("ollama", True, f"TCP reachable at {host}:{port}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")

    llm_provider = os.getenv("LLM_PROVIDER", "auto").lower()
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    db_url = os.getenv("DATABASE_URL", "postgresql://sanius:sanius@localhost:5432/sanius")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    results: list[CheckResult] = []
    results.append(CheckResult("python", True, sys.executable))
    results.append(_check_imports())
    results.append(_check_spacy_models())
    groq_required = llm_provider == "groq"
    groq_ok = bool(groq_api_key) or (not groq_required)
    groq_detail = _mask(groq_api_key) if groq_api_key else ("not set (ok: using ollama)" if not groq_required else "not set")
    results.append(CheckResult("GROQ_API_KEY", groq_ok, groq_detail))
    results.append(_check_ollama(ollama_base_url))

    postgres_result = asyncio.run(_check_postgres(db_url))
    results.append(postgres_result)

    failed = [r for r in results if not r.ok]

    print("Sanius doctor")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"- {status:<4} {r.name}: {r.detail}")

    if failed:
        print("\nNext actions:")
        if any(r.name == "ollama" for r in failed):
            print("- Install Ollama and start it so it listens on port 11434.")
        if any(r.name == "postgres" for r in failed):
            print("- Ensure Postgres+pgvector is running and DATABASE_URL credentials are correct.")
        if any(r.name == "GROQ_API_KEY" for r in failed):
            print("- Either set GROQ_API_KEY (if using Groq) or set LLM_PROVIDER=ollama (to use Ollama for /query).")

        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
