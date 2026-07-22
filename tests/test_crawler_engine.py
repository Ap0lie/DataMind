from __future__ import annotations

from typing import Any

import pytest

from app.crawler.engine import CrawlerEngine
from app.crawler.models import (
    CrawlCacheMode,
    CrawlRequest,
    CrawlResourceType,
    CrawlStatus,
    ExtractedContent,
    FetchedResource,
    RetryOptions,
    RobotsDecision,
)
from app.crawler.strategies import (
    AllowAllRobotsPolicy,
    DefaultCrawlErrorRecoveryPolicy,
    DefaultIncrementalCrawlPolicy,
    HashDuplicateDetector,
    InMemoryCrawlCache,
)


class FakeFetcher:
    def __init__(self, resources: list[FetchedResource | Exception]) -> None:
        self._resources = resources
        self.calls = 0

    async def fetch(self, request: CrawlRequest) -> FetchedResource:
        self.calls += 1
        item = self._resources.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeExtractor:
    async def extract(self, request: CrawlRequest, resource: FetchedResource) -> ExtractedContent:
        text = resource.body.decode("utf-8")
        return ExtractedContent(text=text, raw_content=text, title="Fake title")


class DenyRobotsPolicy:
    async def allowed(self, request: CrawlRequest) -> RobotsDecision:
        return RobotsDecision(allowed=False, reason="Blocked by test robots policy.")


def make_resource(body: str, status_code: int = 200, **headers: Any) -> FetchedResource:
    return FetchedResource(
        url="https://example.com/page",
        status_code=status_code,
        content_type="text/html",
        body=body.encode("utf-8"),
        headers={key.replace("_", "-"): str(value) for key, value in headers.items()},
    )


def make_engine(
    fetcher: FakeFetcher,
    *,
    robots_policy: AllowAllRobotsPolicy | DenyRobotsPolicy | None = None,
    cache: InMemoryCrawlCache | None = None,
    duplicate_detector: HashDuplicateDetector | None = None,
) -> CrawlerEngine:
    return CrawlerEngine(
        fetcher=fetcher,
        extractor=FakeExtractor(),
        robots_policy=robots_policy or AllowAllRobotsPolicy(),
        cache=cache or InMemoryCrawlCache(),
        duplicate_detector=duplicate_detector or HashDuplicateDetector(),
        incremental_policy=DefaultIncrementalCrawlPolicy(),
        recovery_policy=DefaultCrawlErrorRecoveryPolicy(),
    )


@pytest.mark.asyncio
async def test_crawler_fetches_and_extracts_html() -> None:
    fetcher = FakeFetcher([make_resource("<html>Hello DataMind</html>")])
    engine = make_engine(fetcher)

    result = await engine.crawl(CrawlRequest(url="https://example.com/page"))

    assert result.status == CrawlStatus.SUCCESS
    assert result.text == "<html>Hello DataMind</html>"
    assert result.content_hash is not None
    assert fetcher.calls == 1


@pytest.mark.asyncio
async def test_crawler_stops_when_robots_disallows_request() -> None:
    fetcher = FakeFetcher([make_resource("should not fetch")])
    engine = make_engine(fetcher, robots_policy=DenyRobotsPolicy())

    result = await engine.crawl(CrawlRequest(url="https://example.com/page"))

    assert result.status == CrawlStatus.ROBOTS_DISALLOWED
    assert fetcher.calls == 0


@pytest.mark.asyncio
async def test_crawler_uses_fresh_cache_for_incremental_crawl() -> None:
    cache = InMemoryCrawlCache()
    fetcher = FakeFetcher([make_resource("fresh")])
    engine = make_engine(fetcher, cache=cache)
    request = CrawlRequest(url="https://example.com/page")

    first = await engine.crawl(request)
    second = await engine.crawl(request)

    assert first.status == CrawlStatus.SUCCESS
    assert second.status == CrawlStatus.CACHE_HIT
    assert second.from_cache
    assert fetcher.calls == 1


@pytest.mark.asyncio
async def test_crawler_detects_duplicate_content_hash() -> None:
    duplicate_detector = HashDuplicateDetector()
    first_engine = make_engine(
        FakeFetcher([make_resource("same content")]),
        duplicate_detector=duplicate_detector,
    )
    second_engine = make_engine(
        FakeFetcher([make_resource("same content")]),
        duplicate_detector=duplicate_detector,
    )
    no_cache = {"cache": {"mode": CrawlCacheMode.BYPASS}}

    first = await first_engine.crawl(CrawlRequest(url="https://example.com/one", **no_cache))
    second = await second_engine.crawl(CrawlRequest(url="https://example.com/two", **no_cache))

    assert first.status == CrawlStatus.SUCCESS
    assert second.status == CrawlStatus.DUPLICATE
    assert second.duplicate_of is not None


@pytest.mark.asyncio
async def test_crawler_retries_fetch_errors() -> None:
    fetcher = FakeFetcher([RuntimeError("temporary failure"), make_resource("recovered")])
    engine = make_engine(fetcher)

    result = await engine.crawl(
        CrawlRequest(
            url="https://example.com/page",
            retry=RetryOptions(max_retries=1, backoff_seconds=0),
        )
    )

    assert result.status == CrawlStatus.SUCCESS
    assert result.attempts == 2
    assert fetcher.calls == 2


@pytest.mark.asyncio
async def test_crawler_models_support_resource_types() -> None:
    for resource_type in (
        CrawlResourceType.HTML,
        CrawlResourceType.MARKDOWN,
        CrawlResourceType.PDF,
        CrawlResourceType.RSS,
        CrawlResourceType.DYNAMIC,
    ):
        request = CrawlRequest(url="https://example.com/resource", resource_type=resource_type)

        assert request.resource_type == resource_type
