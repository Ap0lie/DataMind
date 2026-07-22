from __future__ import annotations

import asyncio

from app.crawler.contracts import (
    ContentExtractor,
    ContentFetcher,
    CrawlCache,
    CrawlErrorRecoveryPolicy,
    DuplicateDetector,
    IncrementalCrawlPolicy,
    RobotsPolicy,
)
from app.crawler.models import (
    CrawlCacheEntry,
    CrawlCacheMode,
    CrawlError,
    CrawlErrorCode,
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
    FetchedResource,
)
from app.crawler.strategies import content_hash


class CrawlerEngine:
    def __init__(
        self,
        *,
        fetcher: ContentFetcher,
        extractor: ContentExtractor,
        robots_policy: RobotsPolicy,
        cache: CrawlCache,
        duplicate_detector: DuplicateDetector,
        incremental_policy: IncrementalCrawlPolicy,
        recovery_policy: CrawlErrorRecoveryPolicy,
    ) -> None:
        self._fetcher = fetcher
        self._extractor = extractor
        self._robots_policy = robots_policy
        self._cache = cache
        self._duplicate_detector = duplicate_detector
        self._incremental_policy = incremental_policy
        self._recovery_policy = recovery_policy

    async def crawl(self, request: CrawlRequest) -> CrawlResult:
        if request.respect_robots_txt:
            robots_decision = await self._robots_policy.allowed(request)
            if not robots_decision.allowed:
                return self._robots_disallowed(request, robots_decision.reason)

        cache_entry = await self._read_cache(request)
        incremental_decision = await self._incremental_policy.evaluate(request, cache_entry)
        if not incremental_decision.should_fetch and incremental_decision.cached_result is not None:
            return incremental_decision.cached_result

        request_to_fetch = request.model_copy(
            update={"headers": request.headers | incremental_decision.headers}
        )
        resource, attempts, fetch_error = await self._fetch_with_recovery(request_to_fetch)
        if resource is None:
            return self._failed(request, attempts=attempts, error=fetch_error)

        if resource.status_code == 304 and cache_entry is not None:
            return cache_entry.result.model_copy(
                update={
                    "status": CrawlStatus.NOT_MODIFIED,
                    "from_cache": True,
                    "attempts": attempts,
                }
            )

        if resource.status_code >= 400:
            return self._failed(
                request,
                attempts=attempts,
                error=CrawlError(
                    code=CrawlErrorCode.FETCH_FAILED,
                    message=f"Fetch returned HTTP {resource.status_code}.",
                ),
                resource=resource,
            )

        try:
            extracted = await self._extractor.extract(request, resource)
        except Exception as exc:
            return self._failed(
                request,
                attempts=attempts,
                error=CrawlError(
                    code=CrawlErrorCode.EXTRACTION_FAILED,
                    message=str(exc),
                    details={"exception_type": exc.__class__.__name__},
                ),
                resource=resource,
            )

        result = CrawlResult(
            request_id=request.request_id,
            url=request.url,
            final_url=resource.final_url or resource.url,
            resource_type=request.resource_type,
            status=CrawlStatus.SUCCESS,
            status_code=resource.status_code,
            content_type=resource.content_type,
            text=extracted.text,
            raw_content=extracted.raw_content,
            content_hash=content_hash(extracted.text),
            title=extracted.title,
            links=extracted.links,
            metadata=extracted.metadata,
            attempts=attempts,
            fetched_at=resource.fetched_at,
        )

        duplicate_decision = await self._duplicate_detector.check(request, result)
        if duplicate_decision.is_duplicate:
            return result.model_copy(
                update={
                    "status": CrawlStatus.DUPLICATE,
                    "duplicate_of": duplicate_decision.duplicate_of,
                    "error": CrawlError(
                        code=CrawlErrorCode.DUPLICATE,
                        message=duplicate_decision.reason or "Duplicate content detected.",
                    ),
                }
            )

        await self._duplicate_detector.remember(request, result)
        await self._write_cache(request, result, resource)
        return result

    async def _fetch_with_recovery(
        self,
        request: CrawlRequest,
    ) -> tuple[FetchedResource | None, int, CrawlError | None]:
        attempts = 0
        last_error: CrawlError | None = None

        while attempts <= request.retry.max_retries:
            attempts += 1
            try:
                resource = await asyncio.wait_for(
                    self._fetcher.fetch(request),
                    timeout=request.timeout_seconds,
                )
                should_retry = await self._recovery_policy.should_retry(
                    request,
                    attempts,
                    status_code=resource.status_code,
                )
                if should_retry:
                    await self._sleep_before_retry(request, attempts)
                    continue
                return resource, attempts, None
            except TimeoutError:
                last_error = CrawlError(
                    code=CrawlErrorCode.TIMEOUT,
                    message=f"Crawl timed out after {request.timeout_seconds} seconds.",
                )
            except Exception as exc:
                last_error = CrawlError(
                    code=CrawlErrorCode.FETCH_FAILED,
                    message=str(exc),
                    details={"exception_type": exc.__class__.__name__},
                )

            should_retry = await self._recovery_policy.should_retry(
                request,
                attempts,
                error=RuntimeError(last_error.message) if last_error else None,
            )
            if not should_retry:
                break
            await self._sleep_before_retry(request, attempts)

        return None, max(attempts, 1), last_error

    async def _read_cache(self, request: CrawlRequest) -> CrawlCacheEntry | None:
        if request.cache.mode in {CrawlCacheMode.BYPASS, CrawlCacheMode.WRITE_ONLY}:
            return None
        return await self._cache.get(request)

    async def _write_cache(
        self,
        request: CrawlRequest,
        result: CrawlResult,
        resource: FetchedResource,
    ) -> None:
        if request.cache.mode in {CrawlCacheMode.BYPASS, CrawlCacheMode.READ_ONLY}:
            return
        entry = CrawlCacheEntry.from_result(
            request,
            result,
            ttl_seconds=request.cache.ttl_seconds,
            etag=resource.headers.get("etag"),
            last_modified=resource.headers.get("last-modified"),
        )
        await self._cache.set(request, entry)

    async def _sleep_before_retry(self, request: CrawlRequest, attempt: int) -> None:
        delay = await self._recovery_policy.backoff_seconds(request, attempt)
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _robots_disallowed(request: CrawlRequest, reason: str | None) -> CrawlResult:
        return CrawlResult(
            request_id=request.request_id,
            url=request.url,
            resource_type=request.resource_type,
            status=CrawlStatus.ROBOTS_DISALLOWED,
            error=CrawlError(
                code=CrawlErrorCode.ROBOTS_DISALLOWED,
                message=reason or "robots.txt disallowed this crawl request.",
            ),
        )

    @staticmethod
    def _failed(
        request: CrawlRequest,
        *,
        attempts: int,
        error: CrawlError | None,
        resource: FetchedResource | None = None,
    ) -> CrawlResult:
        return CrawlResult(
            request_id=request.request_id,
            url=request.url,
            final_url=resource.final_url if resource else None,
            resource_type=request.resource_type,
            status=CrawlStatus.FAILED,
            status_code=resource.status_code if resource else None,
            content_type=resource.content_type if resource else None,
            error=error
            or CrawlError(code=CrawlErrorCode.UNKNOWN, message="Unknown crawl failure."),
            attempts=attempts,
            fetched_at=resource.fetched_at if resource else None,
        )
