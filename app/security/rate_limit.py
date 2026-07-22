from __future__ import annotations

from functools import lru_cache

from app.core.settings import get_settings


class RateLimitExceeded(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _redis_client(url: str):
    import redis

    return redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)


def enforce_rate_limit(key: str, *, limit: int, window_seconds: int) -> None:
    settings = get_settings()
    if not settings.rate_limits_enabled:
        return
    try:
        client = _redis_client(settings.redis_url)
        normalized_key = f"datamind:rate:{key}"
        with client.pipeline() as pipeline:
            pipeline.incr(normalized_key)
            pipeline.expire(normalized_key, max(1, window_seconds), nx=True)
            count, _ = pipeline.execute()
    except Exception as exc:
        if settings.environment.lower() == "production":
            raise RuntimeError("Rate limit service is unavailable.") from exc
        return
    if int(count) > max(1, limit):
        raise RateLimitExceeded("Too many requests. Please try again later.")
