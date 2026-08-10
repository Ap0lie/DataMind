from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import anyio


class AsyncUpload(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class StagedUpload:
    path: Path
    size_bytes: int

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


async def stage_upload(
    upload: AsyncUpload,
    *,
    staging_root: Path,
    max_bytes: int,
    suffix: str = "",
    chunk_size: int = 1024 * 1024,
) -> StagedUpload:
    """Persist an incoming multipart upload without retaining it in memory."""
    await anyio.to_thread.run_sync(
        lambda: staging_root.mkdir(parents=True, exist_ok=True)
    )
    path = staging_root / f"{uuid4()}{suffix}"
    size = 0
    try:
        async with await anyio.open_file(path, "xb") as target:
            while chunk := await upload.read(chunk_size):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"Uploaded file exceeds the {max_bytes}-byte limit.")
                await target.write(chunk)
        if size == 0:
            raise ValueError("Uploaded file is empty.")
        return StagedUpload(path=path, size_bytes=size)
    except Exception:
        await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))
        raise
