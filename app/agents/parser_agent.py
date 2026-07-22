from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from typing import Any, Protocol

from pydantic import HttpUrl, TypeAdapter

from app.agents.document_models import Document, DocumentFormat, ParsedDocument

HttpUrlAdapter = TypeAdapter(HttpUrl)


class ParserAgentContract(Protocol):
    async def parse(self, document: Document) -> ParsedDocument:
        """Parse, clean, and normalize one crawled document."""


class ParserAgent:
    async def parse(self, document: Document) -> ParsedDocument:
        match document.format:
            case DocumentFormat.HTML:
                parsed = _parse_html(document.content)
            case DocumentFormat.MARKDOWN | DocumentFormat.PDF_TEXT | DocumentFormat.RSS_ITEM:
                parsed = _parse_text_like(document.content)
            case DocumentFormat.PLAIN_TEXT:
                parsed = _parse_text_like(document.content)

        title = document.title or parsed.title
        normalized_text = _normalize_text(parsed.text)
        return ParsedDocument(
            document_id=document.document_id,
            source_url=document.source_url,
            title=title,
            normalized_text=normalized_text,
            metadata=document.metadata | parsed.metadata,
            links=parsed.links,
            content_hash=_content_hash(normalized_text),
        )


class _ParsedContent:
    def __init__(
        self,
        *,
        text: str,
        title: str | None = None,
        links: tuple[HttpUrl, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.title = title
        self.links = links
        self.metadata = metadata or {}


class _HTMLContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.links: list[HttpUrl] = []
        self.metadata: dict[str, Any] = {}
        self.title: str | None = None
        self._current_tag: str | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._current_tag = tag
        attr_map = {key.lower(): value for key, value in attrs if value is not None}
        if tag in {"script", "style", "noscript", "template"}:
            self._skip_depth += 1
        if tag == "a" and "href" in attr_map:
            self._append_url(attr_map["href"])
        if tag == "meta":
            self._append_metadata(attr_map)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "template"} and self._skip_depth > 0:
            self._skip_depth -= 1
        self._current_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        value = data.strip()
        if not value:
            return
        if self._current_tag == "title":
            self.title = value
        self.text_parts.append(value)

    def _append_url(self, value: str) -> None:
        try:
            self.links.append(HttpUrlAdapter.validate_python(value))
        except ValueError:
            return

    def _append_metadata(self, attrs: dict[str, str]) -> None:
        key = attrs.get("name") or attrs.get("property")
        content = attrs.get("content")
        if key and content:
            self.metadata[key] = content


def _parse_html(content: str) -> _ParsedContent:
    parser = _HTMLContentParser()
    parser.feed(content)
    return _ParsedContent(
        text=" ".join(parser.text_parts),
        title=parser.title,
        links=tuple(parser.links),
        metadata=parser.metadata,
    )


def _parse_text_like(content: str) -> _ParsedContent:
    text = re.sub(r"<[^>]+>", " ", content)
    return _ParsedContent(text=text)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
