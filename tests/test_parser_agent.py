from __future__ import annotations

import pytest

from app.agents.document_models import Document, DocumentFormat
from app.agents.parser_agent import ParserAgent


@pytest.mark.asyncio
async def test_parser_agent_extracts_html_metadata_and_normalized_text() -> None:
    document = Document(
        source_url="https://example.com/article",
        format=DocumentFormat.HTML,
        content="""
        <html>
          <head>
            <title>DataMind Runtime</title>
            <meta name="description" content="Enterprise web intelligence" />
            <script>ignore_me()</script>
          </head>
          <body>
            <main><h1>DataMind</h1><p>Clean Architecture for agents.</p></main>
            <a href="https://example.com/ref">Reference</a>
          </body>
        </html>
        """,
    )

    parsed = await ParserAgent().parse(document)

    assert parsed.title == "DataMind Runtime"
    assert parsed.metadata["description"] == "Enterprise web intelligence"
    assert "ignore_me" not in parsed.normalized_text
    assert "Clean Architecture for agents." in parsed.normalized_text
    assert str(parsed.links[0]) == "https://example.com/ref"
    assert parsed.content_hash is not None


@pytest.mark.asyncio
async def test_parser_agent_normalizes_text_like_documents() -> None:
    document = Document(
        source_url="https://example.com/readme.md",
        format=DocumentFormat.MARKDOWN,
        content="# DataMind\n\nA   modular    runtime.",
    )

    parsed = await ParserAgent().parse(document)

    assert parsed.normalized_text == "# DataMind A modular runtime."
