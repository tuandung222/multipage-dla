"""
LLM client utilities for communicating with Gemini via OpenRouter.

Provides a thin wrapper around the OpenAI-compatible API so that
the rest of the pipeline stays model-agnostic.
"""

import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from config import OPENROUTER_BASE_URL, MODEL_NAME

# ── Bootstrap ────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-initialise and return the OpenRouter client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. "
                "Set it in .env or as an environment variable."
            )
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client


# ── Image encoding ───────────────────────────────────────────────

def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data-URL string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def make_image_content(image: Image.Image) -> dict:
    """Wrap a PIL Image into the OpenAI vision message format."""
    return {
        "type": "image_url",
        "image_url": {"url": encode_image(image)},
    }


def make_text_content(text: str) -> dict:
    """Wrap text into the OpenAI message format."""
    return {"type": "text", "text": text}


# ── Chat completion ──────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    *,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> str:
    """Send a chat-completion request and return the assistant text."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
