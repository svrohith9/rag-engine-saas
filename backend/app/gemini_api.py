from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

import requests


def _as_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/jpeg"


def list_models(api_key: str) -> list[str]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = requests.get(url, headers={"x-goog-api-key": api_key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    out: list[str] = []
    for item in data.get("models", []):
        name = item.get("name")
        if isinstance(name, str) and name.startswith("models/"):
            out.append(name.replace("models/", ""))
    return out


def generate_content(
    api_key: str,
    model: str,
    system_text: str,
    user_text: str,
    image_paths: list[str] | None = None,
    temperature: float = 0.2,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    parts: List[Dict[str, Any]] = []
    if user_text:
        parts.append({"text": user_text})

    for p in (image_paths or [])[:3]:
        path = Path(p)
        parts.append(
            {
                "inline_data": {
                    "mime_type": _as_mime(path),
                    "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
                }
            }
        )

    payload: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature},
    }

    resp = requests.post(
        url,
        json=payload,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        timeout=120,
    )

    if resp.status_code == 404:
        raise RuntimeError(f"Gemini 404 for model '{model}'. Use /api/models to find valid model ids.")

    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    out_parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in out_parts if isinstance(part, dict)).strip()
    if not text:
        raise RuntimeError("Gemini returned empty text")
    return text


def embed_text(api_key: str, embed_model: str, text: str) -> list[float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{embed_model}:embedContent"
    payload = {"content": {"parts": [{"text": text}]}}

    resp = requests.post(
        url,
        json=payload,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        timeout=60,
    )

    if resp.status_code == 404:
        raise RuntimeError(
            f"Gemini 404 for embed model '{embed_model}'. Set GEMINI_EMBED_MODEL to an available embedding model."
        )

    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding", {})
    values = emb.get("values")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Embedding response missing 'embedding.values'")
    return [float(x) for x in values]
