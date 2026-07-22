from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

type DataCell = str | int | float | bool | None


class ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetRequest(ToolSchema):
    dataset_id: str | None = None
    records: tuple[dict[str, DataCell], ...] = Field(min_length=1)


class ColumnProfile(ToolSchema):
    name: str = Field(min_length=1)
    inferred_type: str = Field(min_length=1)
    missing_count: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None


class DatasetProfileResponse(ToolSchema):
    dataset_id: str | None = None
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: tuple[ColumnProfile, ...] = Field(default_factory=tuple)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AggregationMetric(ToolSchema):
    column: str = Field(min_length=1)
    operation: str = Field(pattern="^(sum|avg|min|max|count)$")
    alias: str | None = None


class AggregationRequest(DatasetRequest):
    group_by: tuple[str, ...] = Field(default_factory=tuple)
    metrics: tuple[AggregationMetric, ...] = Field(min_length=1)


class AggregationResponse(ToolSchema):
    dataset_id: str | None = None
    rows: tuple[dict[str, DataCell], ...] = Field(default_factory=tuple)


class AnomalyDetectionRequest(DatasetRequest):
    columns: tuple[str, ...] = Field(min_length=1)
    zscore_threshold: float = Field(default=2.0, gt=0)


class AnomalyRecord(ToolSchema):
    row_index: int = Field(ge=0)
    column: str = Field(min_length=1)
    value: float
    zscore: float


class AnomalyDetectionResponse(ToolSchema):
    dataset_id: str | None = None
    anomalies: tuple[AnomalyRecord, ...] = Field(default_factory=tuple)


class NLPToolRequest(ToolSchema):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    backend: str = Field(default="rule_based", pattern="^(rule_based|local_model|llm)$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntitySchema(ToolSchema):
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    canonical_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationSchema(ToolSchema):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeywordSchema(ToolSchema):
    text: str = Field(min_length=1)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class TopicSchema(ToolSchema):
    label: str = Field(min_length=1)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class SummarySchema(ToolSchema):
    text: str = Field(min_length=1)
    compression_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    model: str | None = None


class NERResponse(ToolSchema):
    entities: tuple[EntitySchema, ...] = Field(default_factory=tuple)


class RelationExtractionResponse(ToolSchema):
    relations: tuple[RelationSchema, ...] = Field(default_factory=tuple)


class KeywordExtractionResponse(ToolSchema):
    keywords: tuple[KeywordSchema, ...] = Field(default_factory=tuple)


class TopicClassificationResponse(ToolSchema):
    topics: tuple[TopicSchema, ...] = Field(default_factory=tuple)


class SummarizationResponse(ToolSchema):
    summary: SummarySchema


class LanguageDetectionResponse(ToolSchema):
    language: str = Field(min_length=2)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SentimentAnalysisResponse(ToolSchema):
    sentiment: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelMessage(ToolSchema):
    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str | list[dict[str, Any]] | None = None
    tool_calls: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    tool_call_id: str | None = None


class ModelRouterRequest(ToolSchema):
    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    tool_choice: str | dict[str, Any] | None = None


class ModelRouterResponse(ToolSchema):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    content: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    finish_reason: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
