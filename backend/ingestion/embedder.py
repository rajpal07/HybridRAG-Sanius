import time
import logging
import httpx
from backend.config import OLLAMA_BASE_URL, OLLAMA_API_KEY, EMBED_MODEL

logger = logging.getLogger(__name__)


def _headers() -> dict:
    if OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return {}


def embed_text(text: str) -> list[float]:
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    # Ollama cloud uses /api/embed; local uses /api/embeddings — try both
    url = f"{OLLAMA_BASE_URL}/api/embed"
    embeddings = []

    for text in texts:
        for attempt in range(3):
            try:
                response = httpx.post(
                    url,
                    json={"model": EMBED_MODEL, "input": text},
                    headers=_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                # Cloud returns {"embeddings": [[...]]}; local returns {"embedding": [...]}
                if "embeddings" in data:
                    embeddings.append(data["embeddings"][0])
                else:
                    embeddings.append(data["embedding"])
                break
            except httpx.HTTPStatusError as e:
                # Fall back to legacy /api/embeddings endpoint on 404
                if e.response.status_code == 404 and "embed" in url and attempt == 0:
                    url = f"{OLLAMA_BASE_URL}/api/embeddings"
                    continue
                if attempt < 2:
                    time.sleep(2)
                else:
                    logger.error("Embedding HTTP error: %s", e)
                    raise
            except httpx.ConnectError:
                if attempt == 0:
                    logger.error(
                        "Cannot connect to Ollama at %s. "
                        "Check OLLAMA_BASE_URL and that Ollama is running.",
                        OLLAMA_BASE_URL,
                    )
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Ollama not reachable after 3 attempts at {OLLAMA_BASE_URL}"
                    )
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    logger.error("Embedding failed: %s", e)
                    raise

    return embeddings
