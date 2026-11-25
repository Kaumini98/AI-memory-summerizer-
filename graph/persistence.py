"""PostgreSQL-backed long-term memory storage and connection lifecycle."""

import atexit
import hashlib
import os
import sys
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
MEMORY_CATEGORIES = {"people", "events", "preferences", "important_facts"}
POOL_CONFIG = {"min_size": 1, "max_size": 5}

load_dotenv(dotenv_path=ENV_FILE, override=False)


class MemoryStoreConfigurationError(RuntimeError):
    """Raised when required persistence configuration is missing or invalid."""


class _StoreResource:
    def __init__(self, context: AbstractContextManager[PostgresStore]) -> None:
        self._context = context
        self.store = context.__enter__()
        try:
            self.store.setup()
        except Exception:
            context.__exit__(*sys.exc_info())
            raise

    def close(self) -> None:
        self._context.__exit__(None, None, None)


_store_resource: _StoreResource | None = None
_store_lock = threading.RLock()


def get_database_url() -> str:
    """Return DATABASE_URL without logging or exposing its contents."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise MemoryStoreConfigurationError("DATABASE_URL is not configured.")
    return database_url


def get_user_id(configured_user_id: str | None = None) -> str:
    """Read and validate the configured UUID used to scope local-user memories."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    raw_user_id = configured_user_id or os.environ.get("USER_ID")
    if not raw_user_id:
        raise MemoryStoreConfigurationError("USER_ID is not configured.")
    try:
        return str(UUID(raw_user_id))
    except (ValueError, AttributeError) as exc:
        raise MemoryStoreConfigurationError("USER_ID must be a valid UUID.") from exc


def memories_namespace(user_id: str | UUID | None = None) -> tuple[str, str, str]:
    """Build the server-controlled namespace for one user's long-term memories."""
    validated_user_id = get_user_id(str(user_id)) if user_id is not None else get_user_id()
    return ("users", validated_user_id, "memories")


def memory_key(content: str, category: str) -> str:
    """Build a stable key so repeated extraction upserts the same memory."""
    if not isinstance(category, str) or category not in MEMORY_CATEGORIES:
        raise ValueError("Unsupported memory category.")
    normalized_content = " ".join(content.split()).casefold()
    if not normalized_content:
        raise ValueError("Memory content cannot be empty.")
    key_source = f"{category}\0{normalized_content}".encode("utf-8")
    return hashlib.sha256(key_source).hexdigest()


def get_memory_store() -> PostgresStore:
    """Create the pooled PostgresStore once and reuse it for this process."""
    global _store_resource
    with _store_lock:
        if _store_resource is None:
            context = PostgresStore.from_conn_string(
                get_database_url(), pool_config=POOL_CONFIG
            )
            try:
                _store_resource = _StoreResource(context)
            except Exception:
                raise
        return _store_resource.store


def close_memory_store() -> None:
    """Close the shared connection pool cleanly during process shutdown."""
    global _store_resource
    with _store_lock:
        if _store_resource is not None:
            resource = _store_resource
            _store_resource = None
            resource.close()


atexit.register(close_memory_store)


def save_memories(
    memories: list[dict[str, str]], user_id: str | UUID | None = None
) -> int:
    """Upsert extracted memories in the configured user's namespace."""
    store = get_memory_store()
    namespace = memories_namespace(user_id)
    saved_count = 0

    for memory in memories:
        content = memory.get("content", "")
        category = memory.get("category", "")
        if not isinstance(content, str) or not content.strip():
            continue
        key = memory_key(content, category)
        existing = store.get(namespace, key)
        created_at: Any = None
        if existing is not None and isinstance(existing.value, dict):
            created_at = existing.value.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        store.put(
            namespace,
            key,
            {
                "content": content.strip(),
                "category": category,
                "created_at": created_at,
            },
        )
        saved_count += 1

    return saved_count
