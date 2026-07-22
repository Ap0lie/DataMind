from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class CrawlerModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CrawlResourceType(StrEnum):
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    RSS = "rss"
    DYNAMIC = "dynamic"


class CrawlStatus(StrEnum):
    SUCCESS = "success"
    CACHE_HIT = "cache_hit"
    NOT_MODIFIED = "not_modified"
    DUPLICATE = "duplicate"
    ROBOTS_DISALLOWED = "robots_disallowed"
    FAILED = "failed"


class CrawlErrorCode(StrEnum):
    ROBOTS_DISALLOWED = "robots_disallowed"
    FETCH_FAILED = "fetch_failed"
    EXTRACTION_FAILED = "extraction_failed"
    TIMEOUT = "timeout"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"


class CrawlCacheMode(StrEnum):
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    BYPASS = "bypass"
    REFRESH = "refresh"


class DuplicateScope(StrEnum):
    URL = "url"
    CONTENT_HASH = "content_hash"
    URL_AND_CONTENT_HASH = "url_and_content_hash"


class CrawlError(CrawlerModel):
    code: CrawlErrorCode
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class RetryOptions(CrawlerModel):
    max_retries: int = Field(default=2, ge=0, le=10)
    backoff_seconds: float = Field(default=0.25, ge=0.0)
    retry_on_status_codes: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)


class CacheOptions(CrawlerModel):
    mode: CrawlCacheMode = CrawlCacheMode.READ_WRITE
    ttl_seconds: int = Field(default=3600, ge=0)


class IncrementalOptions(CrawlerModel):
    enabled: bool = True
    use_etag: bool = True
    use_last_modified: bool = True
    force_refresh: bool = False


class DuplicateOptions(CrawlerModel):
    enabled: bool = True
    scope: DuplicateScope = DuplicateScope.CONTENT_HASH


class CrawlRequest(CrawlerModel):
    request_id: UUID = Field(default_factory=uuid4)
    url: HttpUrl
    resource_type: CrawlResourceType = CrawlResourceType.HTML
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0)
    respect_robots_txt: bool = True
    cache: CacheOptions = Field(default_factory=CacheOptions)
    incremental: IncrementalOptions = Field(default_factory=IncrementalOptions)
    duplicate: DuplicateOptions = Field(default_factory=DuplicateOptions)
    retry: RetryOptions = Field(default_factory=RetryOptions)
    metadata: dict[str, str] = Field(default_factory=dict)


class RobotsDecision(CrawlerModel):
    allowed: bool
    reason: str | None = None


class FetchedResource(CrawlerModel):
    url: HttpUrl
    final_url: HttpUrl | None = None
    status_code: int = Field(ge=100, le=599)
    content_type: str = Field(min_length=1)
    body: bytes = Field(default=b"")
    headers: dict[str, str] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("headers")
    @classmethod
    def normalize_headers(cls, value: dict[str, str]) -> dict[str, str]:
        return {key.lower(): header_value for key, header_value in value.items()}


class ExtractedContent(CrawlerModel):
    text: str = ""
    raw_content: str | None = None
    title: str | None = None
    links: tuple[HttpUrl, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrawlCacheEntry(CrawlerModel):
    url: HttpUrl
    resource_type: CrawlResourceType
    result: CrawlResult
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None
    stored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    def is_fresh(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at > (now or datetime.now(UTC))

    @classmethod
    def from_result(
        cls,
        request: CrawlRequest,
        result: CrawlResult,
        *,
        ttl_seconds: int,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> CrawlCacheEntry:
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else now
        return cls(
            url=request.url,
            resource_type=request.resource_type,
            result=result,
            etag=etag,
            last_modified=last_modified,
            content_hash=result.content_hash,
            stored_at=now,
            expires_at=expires_at,
        )


class DuplicateDecision(CrawlerModel):
    is_duplicate: bool
    duplicate_of: HttpUrl | None = None
    reason: str | None = None


class IncrementalDecision(CrawlerModel):
    should_fetch: bool
    cached_result: CrawlResult | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None


class CrawlResult(CrawlerModel):
    request_id: UUID
    url: HttpUrl
    final_url: HttpUrl | None = None
    resource_type: CrawlResourceType
    status: CrawlStatus
    status_code: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = None
    text: str = ""
    raw_content: str | None = None
    content_hash: str | None = None
    title: str | None = None
    links: tuple[HttpUrl, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: CrawlError | None = None
    duplicate_of: HttpUrl | None = None
    attempts: int = Field(default=1, ge=1)
    from_cache: bool = False
    fetched_at: datetime | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
