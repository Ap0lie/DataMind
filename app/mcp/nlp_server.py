from __future__ import annotations

import re
from typing import Protocol

from app.core.enums import McpCapability
from app.mcp.models import MCPServer, MCPTool
from app.mcp.tool_schemas import (
    EntitySchema,
    KeywordExtractionResponse,
    KeywordSchema,
    LanguageDetectionResponse,
    NERResponse,
    NLPToolRequest,
    RelationExtractionResponse,
    RelationSchema,
    SentimentAnalysisResponse,
    SummarizationResponse,
    SummarySchema,
    TopicClassificationResponse,
    TopicSchema,
)


class NLPBackend(Protocol):
    async def ner(self, request: NLPToolRequest) -> NERResponse:
        """Extract named entities."""

    async def relation_extraction(self, request: NLPToolRequest) -> RelationExtractionResponse:
        """Extract relations."""

    async def keyword_extraction(self, request: NLPToolRequest) -> KeywordExtractionResponse:
        """Extract keywords."""

    async def topic_classification(self, request: NLPToolRequest) -> TopicClassificationResponse:
        """Classify topics."""

    async def summarization(self, request: NLPToolRequest) -> SummarizationResponse:
        """Summarize document text."""

    async def language_detection(self, request: NLPToolRequest) -> LanguageDetectionResponse:
        """Detect language."""

    async def sentiment_analysis(self, request: NLPToolRequest) -> SentimentAnalysisResponse:
        """Analyze sentiment."""


class NLPMCPServer:
    def __init__(self, backend: NLPBackend, name: str = "nlp-mcp") -> None:
        self._backend = backend
        self._server = MCPServer(
            name=name,
            description="NLP MCP server for extraction, classification, summary, and analysis.",
            tools=(
                _nlp_tool("ner", "Named entity recognition.", NERResponse.model_json_schema()),
                _nlp_tool(
                    "relation_extraction",
                    "Relation extraction.",
                    RelationExtractionResponse.model_json_schema(),
                ),
                _nlp_tool(
                    "keyword_extraction",
                    "Keyword extraction.",
                    KeywordExtractionResponse.model_json_schema(),
                ),
                _nlp_tool(
                    "topic_classification",
                    "Topic classification.",
                    TopicClassificationResponse.model_json_schema(),
                ),
                _nlp_tool(
                    "summarization",
                    "Document summarization.",
                    SummarizationResponse.model_json_schema(),
                ),
                _nlp_tool(
                    "language_detection",
                    "Language detection.",
                    LanguageDetectionResponse.model_json_schema(),
                ),
                _nlp_tool(
                    "sentiment_analysis",
                    "Sentiment analysis.",
                    SentimentAnalysisResponse.model_json_schema(),
                ),
            ),
        )

    @property
    def server(self) -> MCPServer:
        return self._server

    async def list_tools(self) -> tuple[MCPTool, ...]:
        return self._server.tools

    async def invoke(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        request = NLPToolRequest.model_validate(arguments)
        match tool_name:
            case "ner":
                ner_response = await self._backend.ner(request)
                return ner_response.model_dump(mode="json")
            case "relation_extraction":
                relation_response = await self._backend.relation_extraction(request)
                return relation_response.model_dump(mode="json")
            case "keyword_extraction":
                keyword_response = await self._backend.keyword_extraction(request)
                return keyword_response.model_dump(mode="json")
            case "topic_classification":
                topic_response = await self._backend.topic_classification(request)
                return topic_response.model_dump(mode="json")
            case "summarization":
                summary_response = await self._backend.summarization(request)
                return summary_response.model_dump(mode="json")
            case "language_detection":
                language_response = await self._backend.language_detection(request)
                return language_response.model_dump(mode="json")
            case "sentiment_analysis":
                sentiment_response = await self._backend.sentiment_analysis(request)
                return sentiment_response.model_dump(mode="json")
            case _:
                raise LookupError(f"Unknown NLP MCP tool: {tool_name}")

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        return await self.invoke(tool_name, arguments)


class MockNLPBackend:
    async def ner(self, request: NLPToolRequest) -> NERResponse:
        entities = [
            EntitySchema(
                text=match.group(0),
                label="ORG",
                start_char=match.start(),
                end_char=match.end(),
                confidence=_confidence(request.backend),
            )
            for match in re.finditer(
                r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b",
                request.text,
            )
        ]
        return NERResponse(entities=tuple(entities[:10]))

    async def relation_extraction(self, request: NLPToolRequest) -> RelationExtractionResponse:
        entities = (await self.ner(request)).entities
        if len(entities) < 2:
            return RelationExtractionResponse()
        relation = RelationSchema(
            subject=entities[0].text,
            predicate="related_to",
            object=entities[1].text,
            confidence=_confidence(request.backend) * 0.9,
            evidence=_first_sentence(request.text),
        )
        return RelationExtractionResponse(relations=(relation,))

    async def keyword_extraction(self, request: NLPToolRequest) -> KeywordExtractionResponse:
        words = [
            word.lower()
            for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", request.text)
            if word.lower() not in _STOPWORDS
        ]
        unique_words = list(dict.fromkeys(words))
        keywords = tuple(
            KeywordSchema(text=word, score=max(0.2, 1.0 - index * 0.08))
            for index, word in enumerate(unique_words[:8])
        )
        return KeywordExtractionResponse(keywords=keywords)

    async def topic_classification(self, request: NLPToolRequest) -> TopicClassificationResponse:
        text = request.text.lower()
        if "mcp" in text or "agent" in text:
            topic = "web-intelligence"
        elif "finance" in text:
            topic = "finance"
        else:
            topic = "general"
        return TopicClassificationResponse(
            topics=(TopicSchema(label=topic, score=_confidence(request.backend)),)
        )

    async def summarization(self, request: NLPToolRequest) -> SummarizationResponse:
        sentence = _first_sentence(request.text)
        summary = sentence if len(sentence) <= 240 else f"{sentence[:237]}..."
        ratio = min(1.0, max(0.05, len(summary) / max(len(request.text), 1)))
        return SummarizationResponse(
            summary=SummarySchema(text=summary, compression_ratio=ratio, model=request.backend)
        )

    async def language_detection(self, request: NLPToolRequest) -> LanguageDetectionResponse:
        language = "zh" if any("\u4e00" <= char <= "\u9fff" for char in request.text) else "en"
        return LanguageDetectionResponse(language=language, confidence=_confidence(request.backend))

    async def sentiment_analysis(self, request: NLPToolRequest) -> SentimentAnalysisResponse:
        text = request.text.lower()
        positive = sum(marker in text for marker in ("good", "great", "excellent", "love"))
        negative = sum(marker in text for marker in ("bad", "poor", "hate", "broken"))
        if positive > negative:
            sentiment = "positive"
        elif negative > positive:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return SentimentAnalysisResponse(
            sentiment=sentiment,
            confidence=_confidence(request.backend),
        )


def _nlp_tool(name: str, description: str, output_schema: dict[str, object]) -> MCPTool:
    return MCPTool(
        name=name,
        capability=McpCapability.NLP,
        description=description,
        input_schema=NLPToolRequest.model_json_schema(),
        output_schema=output_schema,
        timeout_seconds=20.0,
        max_retries=1,
        retry_backoff_seconds=0.05,
    )


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts and parts[0] else text.strip()


def _confidence(backend: str) -> float:
    match backend:
        case "llm":
            return 0.9
        case "local_model":
            return 0.82
        case _:
            return 0.7


_STOPWORDS = {
    "about",
    "after",
    "also",
    "from",
    "into",
    "that",
    "their",
    "this",
    "with",
}
