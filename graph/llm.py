"""Shared DeepSeek client configuration for all LLM-backed graph nodes."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DEEPSEEK_MODEL = "deepseek-flash"


@lru_cache(maxsize=1)
def get_deepseek_client() -> OpenAI:
    """Return a process-wide OpenAI-compatible client configured for DeepSeek."""
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        raise RuntimeError(
            f"Set DEEPSEEK_API_KEY to your real API key in {ENV_FILE} before running."
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
