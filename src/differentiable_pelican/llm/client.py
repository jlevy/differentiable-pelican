"""
Shared LLM client utilities with retry logic and robust JSON parsing.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

# Load environment variables from project root .env.local
# Walk up from this file to find the project root (where pyproject.toml lives)
_current = Path(__file__).resolve().parent
for _ancestor in [_current] + list(_current.parents):
    if (_ancestor / "pyproject.toml").exists():
        _env_path = _ancestor / ".env.local"
        if _env_path.exists():
            load_dotenv(dotenv_path=_env_path)
        break

# Centralized model configuration
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def get_client() -> anthropic.Anthropic:
    """
    Get an authenticated Anthropic client.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment. "
            "Set it in .env.local or as an environment variable."
        )
    return anthropic.Anthropic(api_key=api_key)


def extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON from LLM response text, handling common formatting issues.

    Handles:
    - Pure JSON responses
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON with leading/trailing whitespace or text
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object boundaries
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)


def llm_call(
    content: list[dict[str, Any]],
    max_tokens: int = 2048,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    system: str | None = None,
) -> str:
    """
    Make an LLM API call with retry logic.

    Args:
        content: Message content blocks
        max_tokens: Maximum response tokens
        model: Model to use
        max_retries: Number of retries on failure
        system: Optional system prompt

    Returns:
        Response text from the first text content block
    """
    client = get_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        kwargs["system"] = system

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(**kwargs)  # pyright: ignore[reportArgumentType]

            # Extract text from first text content block
            for block in response.content:
                if block.type == "text":
                    return block.text  # type: ignore[attr-defined]

            raise ValueError(f"No text content in response: {response.content}")

        except anthropic.RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            continue

        except anthropic.APIConnectionError as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"Connection error, retrying in {wait}s...")
                time.sleep(wait)
            continue

        except anthropic.APIStatusError as e:
            # Don't retry on 4xx client errors (except rate limit)
            if 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"API error ({e.status_code}), retrying in {wait}s...")
                time.sleep(wait)
            continue

    raise last_error or ValueError("LLM call failed after retries")


def llm_call_json(
    content: list[dict[str, Any]],
    max_tokens: int = 2048,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    system: str | None = None,
) -> dict[str, Any]:
    """
    Make an LLM API call and parse the response as JSON.

    Includes retry logic for both API errors and JSON parse failures.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            text = llm_call(
                content,
                max_tokens=max_tokens,
                model=model,
                max_retries=0,  # Outer loop handles retries
                system=system,
            )
            return extract_json(text)

        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                # On JSON parse failure, add a system reminder and retry
                system = (
                    system or ""
                ) + "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown, no explanatory text."
                print("JSON parse failed, retrying with stricter prompt...")
            continue

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"Error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            continue

    raise last_error or ValueError("LLM JSON call failed after retries")
