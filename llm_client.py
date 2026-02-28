from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict

import ollama
import requests


DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "8"))
OLLAMA_CHAT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "45"))
_LAST_WARNING = ""
_OLLAMA_CLIENT = None


def _set_warning(message: str) -> None:
    global _LAST_WARNING
    _LAST_WARNING = message


def pop_warning() -> str:
    global _LAST_WARNING
    value = _LAST_WARNING
    _LAST_WARNING = ""
    return value


def is_ollama_available(timeout: float = 1.0) -> bool:
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=min(timeout, OLLAMA_TIMEOUT_SECONDS),
        )
        ok = response.status_code == 200
        if not ok:
            _set_warning(f"ollama-unavailable: status={response.status_code} host={OLLAMA_HOST}")
        return ok
    except Exception:
        _set_warning(f"ollama-unavailable: host={OLLAMA_HOST}")
        return False


def _get_ollama_client():
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is None:
        try:
            _OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)
        except Exception:
            _OLLAMA_CLIENT = None
    return _OLLAMA_CLIENT


def _resolve_model_name(model_name: str | None) -> str:
    return (model_name or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL_NAME).strip()


def call_llm(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.1,
    fallback: str = "",
    model_name: str | None = None,
) -> str:
    if not is_ollama_available(timeout=1.2):
        return fallback

    try:
        selected_model = _resolve_model_name(model_name)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _chat():
            client = _get_ollama_client()
            if client is not None:
                return client.chat(
                    model=selected_model,
                    messages=messages,
                    options={"temperature": temperature},
                )
            # Backward-compatible fallback for environments where Client() is unavailable.
            return ollama.chat(
                model=selected_model,
                messages=messages,
                options={"temperature": temperature},
            )

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_chat)
        try:
            response = future.result(timeout=OLLAMA_CHAT_TIMEOUT_SECONDS)
        finally:
            # Do not wait for the worker when timed out; return control immediately.
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        return response["message"]["content"]
    except FuturesTimeoutError:
        _set_warning(f"ollama-timeout: model={selected_model} chat>{OLLAMA_CHAT_TIMEOUT_SECONDS}s")
        return fallback
    except Exception as exc:
        _set_warning(f"ollama-call-failed: {type(exc).__name__}: {exc}")
        return fallback


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def call_llm_json(
    prompt: str,
    schema_hint: str,
    fallback: Dict[str, Any],
    model_name: str | None = None,
) -> Dict[str, Any]:
    system_prompt = (
        "Return only valid JSON. Do not wrap in markdown. "
        f"Schema hint: {schema_hint}"
    )
    text = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.0,
        model_name=model_name,
    )
    parsed = extract_json(text)
    if not parsed:
        return fallback
    merged = dict(fallback)
    merged.update(parsed)
    return merged
