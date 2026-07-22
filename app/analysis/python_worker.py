from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from app.analysis.cleaning_sandbox import CLEANING_WORKER_SOURCE
from app.analysis.python_sandbox import _WORKER_SOURCE


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.analysis.python_worker /input/payload.json")
    payload_path = Path(sys.argv[1])
    payload_text = payload_path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    worker_source = CLEANING_WORKER_SOURCE if payload.get("execution_kind") == "cleaning" else _WORKER_SOURCE
    with io.StringIO(payload_text) as payload_file:
        original_stdin = sys.stdin
        try:
            sys.stdin = payload_file
            namespace: dict[str, object] = {}
            exec(
                compile(worker_source, "<datamind-container-worker>", "exec"),
                namespace,
                namespace,
            )
        finally:
            sys.stdin = original_stdin


if __name__ == "__main__":
    main()
