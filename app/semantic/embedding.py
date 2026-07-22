from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Protocol

from app.core.settings import Settings, get_settings


class SemanticEmbeddingProvider(Protocol):
    model_revision: str

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...
    def status(self) -> str: ...


class DisabledEmbeddingProvider:
    model_revision = "disabled"

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(() for _ in texts)

    def status(self) -> str:
        return "disabled"


class MockEmbeddingProvider:
    model_revision = "mock-v1"

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = []
        for text in texts:
            normalized = _normalize(text)
            values = [float(normalized.count(token)) for token in ("销售", "收入", "地区", "客户", "产品", "日期")]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append(tuple(value / norm for value in values))
        return tuple(vectors)

    def status(self) -> str:
        return "ready"


class PersistentEmbeddingProvider:
    def __init__(self, provider: SemanticEmbeddingProvider, repository: object) -> None:
        self.provider = provider
        self.repository = repository
        self.model_revision = provider.model_revision

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        hashes = tuple(hashlib.sha256(text.encode()).hexdigest() for text in texts)
        cached = self.repository.get_semantic_embedding_cache(model_revision=self.model_revision, text_hashes=hashes)
        missing_indexes = [index for index, text_hash in enumerate(hashes) if text_hash not in cached]
        if missing_indexes:
            computed = self.provider.encode([texts[index] for index in missing_indexes])
            additions = {hashes[index]: vector for index, vector in zip(missing_indexes, computed, strict=True)}
            if any(additions.values()):
                self.repository.save_semantic_embedding_cache(model_revision=self.model_revision, vectors=additions)
                cached.update(additions)
        return tuple(cached.get(text_hash, ()) for text_hash in hashes)

    def status(self) -> str:
        return self.provider.status()


class SentenceTransformerEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_revision = settings.semantic_embedding_revision
        self._model = None
        self._lock = Lock()
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            model = self._load()
            vectors = model.encode(
                missing,
                batch_size=self._settings.semantic_embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = tuple(float(value) for value in vector)
                self._cache.move_to_end(text)
            while len(self._cache) > self._settings.semantic_embedding_cache_size:
                self._cache.popitem(last=False)
        return tuple(self._cache[text] for text in texts)

    def status(self) -> str:
        try:
            self._load()
            return "ready"
        except Exception:
            return "failed" if self._settings.semantic_embedding_required else "fallback"

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            model_path = Path(self._settings.semantic_embedding_model_path)
            source = str(model_path) if model_path.exists() else self._settings.semantic_embedding_model
            self._model = SentenceTransformer(
                source,
                device=self._settings.semantic_embedding_device,
                local_files_only=self._settings.semantic_embedding_local_files_only,
            )
            return self._model


_provider: SemanticEmbeddingProvider | None = None
_provider_key: tuple[object, ...] | None = None
_provider_lock = Lock()


def get_semantic_embedding_provider(settings: Settings | None = None) -> SemanticEmbeddingProvider:
    global _provider, _provider_key
    resolved = settings or get_settings()
    key = (
        resolved.semantic_embedding_enabled,
        resolved.semantic_embedding_model,
        resolved.semantic_embedding_model_path,
        resolved.semantic_embedding_revision,
        resolved.semantic_embedding_device,
    )
    with _provider_lock:
        if _provider is not None and key == _provider_key:
            return _provider
        _provider = SentenceTransformerEmbeddingProvider(resolved) if resolved.semantic_embedding_enabled else DisabledEmbeddingProvider()
        _provider_key = key
        return _provider


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


def _normalize(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())
