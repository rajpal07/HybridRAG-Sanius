import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx

from backend.config import (
    GROQ_API_KEY,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    LLM_PROVIDER,
    LLM_FALLBACK_PROVIDER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq circuit breaker — once rate-limited, skip Groq for GROQ_COOLDOWN_SECONDS
# so every subsequent pipeline call doesn't hammer Groq and waste time on 429s.
# ---------------------------------------------------------------------------
_groq_rate_limited_until: float = 0.0  # epoch seconds
GROQ_COOLDOWN_SECONDS = 60


def _groq_is_cooling_down() -> bool:
    return time.time() < _groq_rate_limited_until


def _set_groq_cooldown() -> None:
    global _groq_rate_limited_until
    _groq_rate_limited_until = time.time() + GROQ_COOLDOWN_SECONDS
    logger.warning(
        "Groq circuit breaker OPEN — skipping Groq for next %ds", GROQ_COOLDOWN_SECONDS
    )


def _resolve_primary_provider() -> str:
    if LLM_PROVIDER in ("auto", "", None):
        return "groq" if GROQ_API_KEY else "ollama"
    return LLM_PROVIDER


def _should_try_fallback(primary: str) -> bool:
    if not LLM_FALLBACK_PROVIDER or LLM_FALLBACK_PROVIDER in ("off", "none", "false", "0"):
        return False
    if LLM_FALLBACK_PROVIDER == primary:
        return False
    return True


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    # Covers: HTTP 429, "rate limit", provider-specific messages
    return ("429" in msg) or ("rate limit" in msg) or ("rate-limit" in msg)


def _is_context_length_error(err: Exception) -> bool:
    msg = str(err).lower()
    # Groq returns 400 when prompt tokens exceed the model's context window.
    # Also catches 413 (payload too large) from any provider.
    return (
        ("400" in msg and ("token" in msg or "context" in msg or "length" in msg))
        or "413" in msg
        or "context_length_exceeded" in msg
        or "too many tokens" in msg
        or "maximum context length" in msg
        or "reduce the length" in msg
    )


def _ollama_headers() -> dict:
    # Ollama local ignores auth; cloud gateways may require it.
    if OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return {}


async def _ollama_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    # Prefer /api/chat; fall back to /api/generate when /api/chat is not available.
    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                r = await client.post(chat_url, json=payload, headers=_ollama_headers())
                if r.status_code == 404:
                    gen_url = f"{OLLAMA_BASE_URL}/api/generate"
                    gen_payload = {
                        "model": model,
                        "prompt": "\n\n".join(m.get("content", "") for m in messages),
                        "stream": False,
                        "options": {"temperature": temperature, "num_predict": max_tokens},
                    }
                    r = await client.post(gen_url, json=gen_payload, headers=_ollama_headers())
                r.raise_for_status()
                data = r.json()
                if "message" in data and isinstance(data["message"], dict):
                    text = (data["message"].get("content") or "").strip()
                else:
                    text = (data.get("response") or "").strip()
                if text:
                    return text

                # Some models/providers occasionally return an empty chat message. Fall back to /api/generate.
                gen_url = f"{OLLAMA_BASE_URL}/api/generate"
                gen_payload = {
                    "model": model,
                    "prompt": "\n\n".join(m.get("content", "") for m in messages),
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                }
                r2 = await client.post(gen_url, json=gen_payload, headers=_ollama_headers())
                r2.raise_for_status()
                data2 = r2.json()
                text2 = (data2.get("response") or "").strip()
                if not text2:
                    raise RuntimeError("Ollama returned an empty response")
                return text2
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 4:
                delay = 2 if not _is_rate_limit_error(e) else min(20, 4 * (attempt + 1))
                await asyncio.sleep(delay)
            else:
                raise
    if last_err:
        raise last_err
    raise RuntimeError("Ollama chat failed without an exception (unexpected).")


def _groq_chat_sync(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    from groq import Groq  # local import so Ollama-only installs still work

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


async def chat(
    *,
    model: str,
    ollama_model: Optional[str] = None,
    messages: list[dict[str, str]],
    max_tokens: int = 1500,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> str:
    primary = (provider or _resolve_primary_provider()).lower()

    if primary == "ollama":
        use_model = ollama_model or model
        return await _ollama_chat(model=use_model, messages=messages, max_tokens=max_tokens, temperature=temperature)

    if primary == "groq":
        # Circuit breaker: if Groq is cooling down, go straight to fallback
        if _groq_is_cooling_down() and _should_try_fallback(primary):
            logger.info("Groq circuit breaker CLOSED — routing directly to %s", LLM_FALLBACK_PROVIDER)
            return await chat(
                model=model,
                ollama_model=ollama_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=LLM_FALLBACK_PROVIDER,
            )
        try:
            return await asyncio.to_thread(
                _groq_chat_sync,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            should_fallback = _should_try_fallback(primary) and (
                _is_rate_limit_error(e) or _is_context_length_error(e)
            )
            if should_fallback:
                if _is_rate_limit_error(e):
                    _set_groq_cooldown()  # open the circuit breaker
                reason = "context length exceeded" if _is_context_length_error(e) else "rate limited"
                logger.warning("Groq %s; falling back to %s", reason, LLM_FALLBACK_PROVIDER)
                return await chat(
                    model=model,
                    ollama_model=ollama_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    provider=LLM_FALLBACK_PROVIDER,
                )
            raise

    raise ValueError(f"Unsupported LLM provider: {primary}")


def _strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    parts = raw.split("```")
    if len(parts) < 2:
        return raw
    fenced = parts[1].strip()
    # ```json ... or ```sql ...
    if "\n" in fenced:
        first, rest = fenced.split("\n", 1)
        if first.strip().lower() in ("json", "sql"):
            return rest.strip()
    return fenced.strip()


async def chat_json(
    *,
    model: str,
    ollama_model: Optional[str] = None,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1500,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> Any:
    raw = await chat(
        model=model,
        ollama_model=ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
    )
    raw = _strip_code_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_obj = raw.find("{")
        end_obj = raw.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            return json.loads(raw[start_obj : end_obj + 1])
        start_arr = raw.find("[")
        end_arr = raw.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return json.loads(raw[start_arr : end_arr + 1])
        raise
