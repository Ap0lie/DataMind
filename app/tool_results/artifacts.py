from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolArtifactTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class ArchivedPayload:
    payload_sha256: str
    size_bytes: int
    compressed_size_bytes: int
    storage_path: str


def archive_json_payload(
    payload: Any,
    *,
    root: Path,
    max_bytes: int,
) -> ArchivedPayload:
    """Stream JSON through gzip while hashing the uncompressed representation."""

    root.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256()
    size = 0
    handle = tempfile.NamedTemporaryFile(prefix="tool-", suffix=".tmp", dir=root, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with gzip.open(temporary, "wb", compresslevel=6) as stream:
            for fragment in encoder.iterencode(payload):
                encoded = fragment.encode("utf-8")
                size += len(encoded)
                if size > max_bytes:
                    raise ToolArtifactTooLarge(
                        f"Tool result exceeds the {max_bytes}-byte archival limit."
                    )
                digest.update(encoded)
                stream.write(encoded)
        payload_sha256 = digest.hexdigest()
        relative = Path(payload_sha256[:2]) / f"{payload_sha256}.json.gz"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, target)
        return ArchivedPayload(
            payload_sha256=payload_sha256,
            size_bytes=size,
            compressed_size_bytes=target.stat().st_size,
            storage_path=relative.as_posix(),
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_json_payload(*, root: Path, storage_path: str) -> Any:
    root_resolved = root.resolve()
    target = (root / storage_path).resolve()
    if not target.is_relative_to(root_resolved):
        raise ValueError("Tool artifact path is outside the configured store.")
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        return json.load(stream)
