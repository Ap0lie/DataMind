from __future__ import annotations

import pytest

from app.agents.document_models import NLPBackendKind, ParsedDocument
from app.agents.nlp_agent import NLPAgent
from app.core.enums import McpCapability
from app.mcp.nlp_server import MockNLPBackend, NLPMCPServer
from app.mcp.runtime import InMemoryMCPRuntime


@pytest.mark.asyncio
async def test_nlp_mcp_registers_all_nlp_tools() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(NLPMCPServer(MockNLPBackend()))

    registry = await runtime.discover(McpCapability.NLP)

    assert {tool.name for tool in registry.tools} == {
        "ner",
        "relation_extraction",
        "keyword_extraction",
        "topic_classification",
        "summarization",
        "language_detection",
        "sentiment_analysis",
    }


@pytest.mark.asyncio
async def test_nlp_agent_invokes_all_nlp_mcp_capabilities() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(NLPMCPServer(MockNLPBackend()))
    document = ParsedDocument(
        document_id="00000000-0000-0000-0000-000000000001",
        source_url="https://example.com/article",
        title="DataMind MCP",
        normalized_text=(
            "DataMind uses MCP and Agent Runtime. "
            "DeepSeek provides excellent model access."
        ),
        metadata={"source": "test"},
    )

    extraction = await NLPAgent(runtime).analyze(document, backend=NLPBackendKind.LOCAL_MODEL)

    assert extraction.backend == NLPBackendKind.LOCAL_MODEL
    assert extraction.entities
    assert extraction.relations
    assert extraction.keywords
    assert extraction.topics[0].label == "web-intelligence"
    assert extraction.summary is not None
    assert extraction.language is not None
    assert extraction.sentiment is not None
    assert extraction.sentiment.sentiment == "positive"


@pytest.mark.asyncio
async def test_nlp_agent_outputs_knowledge_ready_document() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(NLPMCPServer(MockNLPBackend()))
    document = ParsedDocument(
        document_id="00000000-0000-0000-0000-000000000002",
        source_url="https://example.com/article",
        normalized_text="DataMind is an enterprise Web Intelligence Runtime.",
    )

    payload = await NLPAgent(runtime).prepare_for_knowledge_agent(document)

    assert payload.document == document
    assert payload.extraction.document_id == document.document_id
    assert payload.source_url == document.source_url
