import os
from pathlib import Path
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sanius:sanius@localhost:5432/sanius")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
EHG_SOURCE_DIR = os.getenv("EHG_SOURCE_DIR", "")
MIMIC_SOURCE_DIR = os.getenv(
    "MIMIC_SOURCE_DIR",
    str(
        _REPO_ROOT
        / "data"
        / "raw"
        / "mimiciii_demo"
        / "unzipped"
        / "mimic-iii-clinical-database-demo-1.4"
    ),
)
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_lg")
NOTES_PROVIDER = os.getenv("NOTES_PROVIDER", "groq").lower()  # groq|ollama
NOTES_MODEL = os.getenv("NOTES_MODEL", "qwen3.5:latest")
NOTES_FAST_MODE = os.getenv("NOTES_FAST_MODE", "false").lower() in ("1", "true", "yes")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # auto|groq|ollama
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "ollama").lower()  # groq|ollama|off

# Model names are provider-specific. For Groq, keep the existing defaults.
# For Ollama, set MAIN_MODEL/ROUTER_MODEL in .env to a model you have pulled (or a cloud model you have access to).
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")
MAIN_MODEL = os.getenv("MAIN_MODEL", "llama-3.3-70b-versatile")
OLLAMA_ROUTER_MODEL = os.getenv("OLLAMA_ROUTER_MODEL", NOTES_MODEL)
OLLAMA_MAIN_MODEL = os.getenv("OLLAMA_MAIN_MODEL", NOTES_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
