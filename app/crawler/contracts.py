from __future__ import annotations

from typing import Protocol

from app.crawler.models import (
    CrawlCacheEntry,
    CrawlRequest,
    CrawlResult,
    DuplicateDecision,
    ExtractedContent,
    FetchedResource,
    IncrementalDecision,
    RobotsDecision,
)


class Crawler(Protocol):
    async def crawl(self, request: CrawlRequest) -> CrawlResult:
        """Crawl one resource and return a standardized result."""


class ContentFetcher(Protocol):
    async def fetch(self, request: CrawlRequest) -> FetchedResource:
        """Fetch content through an MCP-backed adapter."""


class ContentExtractor(Protocol):
    async def extract(self, request: CrawlRequest, resource: FetchedResource) -> ExtractedContent:
        """Extract text and metadata from fetched HTML, Markdown, PDF, RSS, or dynamic pages."""


class RobotsPolicy(Protocol):
    async def allowed(self, request: CrawlRequest) -> RobotsDecision:
        """Check robots.txt policy before fetch."""


class CrawlCache(Protocol):
    async def get(self, request: CrawlRequest) -> CrawlCacheEntry | None:
        """Return a cache entry for a request if available."""

    async def set(self, request: CrawlRequest, entry: CrawlCacheEntry) -> None:
        """Store a cache entry."""


class DuplicateDetector(Protocol):
    async def check(self, request: CrawlRequest, result: CrawlResult) -> DuplicateDecision:
        """Check whether the crawl result duplicates an existing resource."""

    async def remember(self, request: CrawlRequest, result: CrawlResult) -> None:
        """Record a non-duplicate result for future detection."""


class IncrementalCrawlPolicy(Protocol):
    async def evaluate(
        self,
        request: CrawlRequest,
        cache_entry: CrawlCacheEntry | None,
    ) -> IncrementalDecision:
        """Decide whether a request should be fetched or satisfied from cache."""


class CrawlErrorRecoveryPolicy(Protocol):
    async def should_retry(
        self,
        request: CrawlRequest,
        attempt: int,
        error: Exception | None = None,
        status_code: int | None = None,
    ) -> bool:
        """Decide if a failed fetch/extract attempt should retry."""

    async def backoff_seconds(self, request: CrawlRequest, attempt: int) -> float:
        """Return retry backoff for one attempt."""
