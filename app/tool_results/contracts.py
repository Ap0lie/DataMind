from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ToolResultKind(StrEnum):
    SQL = "sql"
    TABLE = "table"
    REPORT = "report"
    PYTHON = "python"
    ERROR = "error"
    JSON = "json"
    TEXT = "text"


class ToolResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ToolResultEnvelope(BaseModel):
    """Validated execution result before archival and model-facing reduction."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: UUID
    tool_name: str = Field(min_length=1)
    action_hash: str = Field(min_length=1)
    status: ToolResultStatus = ToolResultStatus.SUCCEEDED
    payload: Any
    kind: ToolResultKind | None = None
    content_type: str = "application/json"
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalFact(BaseModel):
    path: str
    value: Any
    value_type: str
    evidence_id: str | None = None


class ToolResultArtifact(BaseModel):
    artifact_id: UUID
    run_id: UUID
    tool_name: str
    action_hash: str
    payload_sha256: str
    status: ToolResultStatus
    kind: ToolResultKind
    content_type: str
    size_bytes: int = Field(ge=0)
    compressed_size_bytes: int = Field(ge=0)
    storage_path: str
    expires_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ToolResultSummary(BaseModel):
    summary_version: int = Field(default=1, ge=1)
    artifact_id: UUID | None = None
    tool_name: str
    kind: ToolResultKind
    status: ToolResultStatus
    headline: str
    schema_fields: tuple[str, ...] = ()
    row_count: int | None = Field(default=None, ge=0)
    canonical_facts: tuple[CanonicalFact, ...] = ()
    preview: tuple[dict[str, Any], ...] = ()
    key_findings: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    validation_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
    omitted_sections: tuple[str, ...] = ()
    deterministic: bool = True
    verified: bool = False


class ToolResultChunk(BaseModel):
    chunk_index: int = Field(ge=0)
    section: str
    content: str
    content_sha256: str
    size_bytes: int = Field(ge=0)


class ToolResultChunkSummary(BaseModel):
    chunk_index: int = Field(ge=0)
    section: str
    content_sha256: str
    summary: str = ""
    source_quotes: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    verified: bool = False
    verification_issues: tuple[str, ...] = ()


class DistillationAttempt(BaseModel):
    attempt: int = Field(ge=1)
    provider: str | None = None
    model: str | None = None
    status: str
    error: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ToolContextBundle(BaseModel):
    artifact_id: UUID
    summary: ToolResultSummary
    model_content: str
    original_size_bytes: int = Field(ge=0)
    context_size_bytes: int = Field(ge=0)
    reduction_ratio: float = Field(ge=0.0, le=1.0)
    distillation_attempts: tuple[DistillationAttempt, ...] = ()


class ToolResultDistillationResult(BaseModel):
    summary: ToolResultSummary
    chunks: tuple[ToolResultChunkSummary, ...] = ()
    attempts: tuple[DistillationAttempt, ...] = ()
    provider: str | None = None
    model: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ToolResultExcerpt(BaseModel):
    path: str
    text: str
    source: str
    score: float = Field(ge=0.0)


class ToolResultProjection(BaseModel):
    """Bounded, source-extractive continuation for one archived tool result."""

    projection_id: UUID | None = None
    artifact_id: UUID
    query_hash: str
    headline: str
    excerpts: tuple[ToolResultExcerpt, ...] = ()
    canonical_facts: tuple[CanonicalFact, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    selected_paths: tuple[str, ...] = ()
    scanned_bytes: int = Field(default=0, ge=0)
    context_size_bytes: int = Field(default=0, ge=0)
    truncated: bool = False
    more_available: bool = False
    verified: bool = True

    def model_context(self) -> dict[str, Any]:
        """Return only source-backed content that is useful to the main model."""

        return {
            "headline": self.headline,
            "excerpts": [item.model_dump(mode="json") for item in self.excerpts],
            "canonical_facts": [
                item.model_dump(mode="json") for item in self.canonical_facts
            ],
            "evidence_ids": list(self.evidence_ids),
            "truncated": self.truncated,
            "more_available": self.more_available,
            "verified": self.verified,
        }
