from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from app.crawler.models import (
    CrawlCacheEntry,
    CrawlCacheMode,
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
    DuplicateDecision,
    DuplicateScope,
    IncrementalDecision,
    RobotsDecision,
)


class AllowAllRobotsPolicy:
    async def allowed(self, request: CrawlRequest) -> RobotsDecision:
        return RobotsDecision(allowed=True)


class InMemoryCrawlCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CrawlCacheEntry] = {}

    async def get(self, request: CrawlRequest) -> CrawlCacheEntry | None:
        return self._entries.get(self._key(request))

    async def set(self, request: CrawlRequest, entry: CrawlCacheEntry) -> None:
        self._entries[self._key(request)] = entry

    @staticmethod
    def _key(request: CrawlRequest) -> tuple[str, str]:
        return (str(request.url), request.resource_type.value)


class HashDuplicateDetector:
    def __init__(self) -> None:
        self._by_url: dict[str, str] = {}
        self._by_hash: dict[str, str] = {}

    async def check(self, request: CrawlRequest, result: CrawlResult) -> DuplicateDecision:
        if not request.duplicate.enabled:
            return DuplicateDecision(is_duplicate=False)

        url_key = str(result.final_url or result.url)
        hash_key = result.content_hash
        scope = request.duplicate.scope

        if scope in {DuplicateScope.URL, DuplicateScope.URL_AND_CONTENT_HASH}:
            original_url = self._by_url.get(url_key)
            if original_url is not None:
                return DuplicateDecision(
                    is_duplicate=True,
                    duplicate_of=original_url,
                    reason="URL has already been crawled.",
                )

        if hash_key is not None and scope in {
            DuplicateScope.CONTENT_HASH,
            DuplicateScope.URL_AND_CONTENT_HASH,
        }:
            original_url = self._by_hash.get(hash_key)
            if original_url is not None:
                return DuplicateDecision(
                    is_duplicate=True,
                    duplicate_of=original_url,
                    reason="Content hash has already been crawled.",
                )

        return DuplicateDecision(is_duplicate=False)

    async def remember(self, request: CrawlRequest, result: CrawlResult) -> None:
        url_key = str(result.final_url or result.url)
        self._by_url[url_key] = url_key
        if result.content_hash is not None:
            self._by_hash[result.content_hash] = url_key


class DefaultIncrementalCrawlPolicy:
    async def evaluate(
        self,
        request: CrawlRequest,
        cache_entry: CrawlCacheEntry | None,
    ) -> IncrementalDecision:
        if cache_entry is None or not request.incremental.enabled:
            return IncrementalDecision(should_fetch=True)

        if request.incremental.force_refresh or request.cache.mode == CrawlCacheMode.REFRESH:
            return IncrementalDecision(should_fetch=True, reason="Refresh requested.")

        if request.cache.mode in {
            CrawlCacheMode.READ_WRITE,
            CrawlCacheMode.READ_ONLY,
        } and cache_entry.is_fresh(datetime.now(UTC)):
            cached_result = cache_entry.result.model_copy(
                update={"status": CrawlStatus.CACHE_HIT, "from_cache": True}
            )
            return IncrementalDecision(
                should_fetch=False,
                cached_result=cached_result,
                reason="Fresh cache entry found.",
            )

        headers: dict[str, str] = {}
        if request.incremental.use_etag and cache_entry.etag:
            headers["if-none-match"] = cache_entry.etag
        if request.incremental.use_last_modified and cache_entry.last_modified:
            headers["if-modified-since"] = cache_entry.last_modified

        return IncrementalDecision(
            should_fetch=True,
            headers=headers,
            reason="Conditional fetch required." if headers else None,
        )


class DefaultCrawlErrorRecoveryPolicy:
    async def should_retry(
        self,
        request: CrawlRequest,
        attempt: int,
        error: Exception | None = None,
        status_code: int | None = None,
    ) -> bool:
        if attempt > request.retry.max_retries:
            return False
        if error is not None:
            return True
        return status_code in request.retry.retry_on_status_codes

    async def backoff_seconds(self, request: CrawlRequest, attempt: int) -> float:
        return request.retry.backoff_seconds * max(attempt, 1)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
