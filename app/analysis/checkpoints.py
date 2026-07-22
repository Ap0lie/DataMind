from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.settings import Settings, get_settings

_checkpointer: Any | None = None
_checkpointer_key: tuple[str, str | None, str] | None = None
_lock = Lock()


def get_analysis_checkpointer(
    settings: Settings | None = None,
    *,
    dataset_store_path: str | None = None,
) -> Any | None:
    global _checkpointer, _checkpointer_key
    resolved = settings or get_settings()
    backend = resolved.checkpoint_backend.lower()
    store_path = dataset_store_path or resolved.dataset_store_path
    key = (backend, resolved.database_url, store_path)
    if _checkpointer is not None and _checkpointer_key == key:
        return _checkpointer
    with _lock:
        if _checkpointer is not None and _checkpointer_key == key:
            return _checkpointer
        try:
            if backend == "postgres":
                if not resolved.database_url:
                    raise RuntimeError("Postgres checkpointing requires DATAMIND_DATABASE_URL.")
                import psycopg
                from langgraph.checkpoint.postgres import PostgresSaver

                psycopg_url = resolved.database_url.replace(
                    "postgresql+psycopg://",
                    "postgresql://",
                    1,
                )
                connection = psycopg.Connection.connect(
                    psycopg_url,
                    autocommit=True,
                    prepare_threshold=0,
                )
                checkpointer = PostgresSaver(connection)
                checkpointer.setup()
            elif backend == "sqlite":
                from langgraph.checkpoint.sqlite import SqliteSaver

                root = Path(store_path)
                checkpoint_path = (
                    root.parent / "datamind_checkpoints.db"
                    if root.name == "datasets"
                    else root / "datamind_checkpoints.db"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
                checkpointer = SqliteSaver(connection)
            else:
                return None
        except ImportError:
            return None
        _checkpointer = checkpointer
        _checkpointer_key = key
        return checkpointer
