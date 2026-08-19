from __future__ import annotations

import hashlib

from app.tool_results.contracts import ToolResultChunk


def chunk_text(
    content: str,
    *,
    section: str = "payload",
    max_chars: int = 12_000,
    max_chunks: int | None = None,
) -> tuple[ToolResultChunk, ...]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    chunks: list[ToolResultChunk] = []
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max_chunks must be positive.")
    starts = range(0, len(content), max_chars)
    for index, start in enumerate(starts):
        if max_chunks is not None and index >= max_chunks:
            break
        text = content[start : start + max_chars]
        chunks.append(
            ToolResultChunk(
                chunk_index=index,
                section=section,
                content=text,
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                size_bytes=len(text.encode("utf-8")),
            )
        )
    return tuple(chunks)
