from __future__ import annotations

from typing import Any

from app.agents.document_models import (
    Entity,
    Keyword,
    KnowledgeReadyDocument,
    LanguageDetection,
    NLPBackendKind,
    NLPExtraction,
    ParsedDocument,
    Relation,
    SentimentAnalysis,
    Summary,
    Topic,
)
from app.mcp.contracts import MCPRuntime


class NLPAgent:
    def __init__(
        self,
        runtime: MCPRuntime,
        *,
        backend: NLPBackendKind = NLPBackendKind.RULE_BASED,
        server_name: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._backend = backend
        self._server_name = server_name

    async def analyze(
        self,
        document: ParsedDocument,
        *,
        backend: NLPBackendKind | None = None,
    ) -> NLPExtraction:
        selected_backend = backend or self._backend
        payload = {
            "document_id": str(document.document_id),
            "text": document.normalized_text,
            "backend": selected_backend.value,
            "metadata": document.metadata,
        }

        entities_data = await self._invoke("ner", payload)
        relations_data = await self._invoke("relation_extraction", payload)
        keywords_data = await self._invoke("keyword_extraction", payload)
        topics_data = await self._invoke("topic_classification", payload)
        summary_data = await self._invoke("summarization", payload)
        language_data = await self._invoke("language_detection", payload)
        sentiment_data = await self._invoke("sentiment_analysis", payload)

        return NLPExtraction(
            document_id=document.document_id,
            source_url=document.source_url,
            backend=selected_backend,
            entities=tuple(Entity.model_validate(item) for item in entities_data["entities"]),
            relations=tuple(Relation.model_validate(item) for item in relations_data["relations"]),
            keywords=tuple(Keyword.model_validate(item) for item in keywords_data["keywords"]),
            topics=tuple(Topic.model_validate(item) for item in topics_data["topics"]),
            summary=Summary.model_validate(summary_data["summary"]),
            language=LanguageDetection.model_validate(language_data),
            sentiment=SentimentAnalysis.model_validate(sentiment_data),
        )

    async def prepare_for_knowledge_agent(
        self,
        document: ParsedDocument,
        *,
        backend: NLPBackendKind | None = None,
    ) -> KnowledgeReadyDocument:
        extraction = await self.analyze(document, backend=backend)
        return KnowledgeReadyDocument(document=document, extraction=extraction)

    async def _invoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self._runtime.invoke(
            tool_name,
            payload,
            server_name=self._server_name,
        )
        if not result.ok:
            message = result.error.message if result.error else "NLP MCP invocation failed."
            raise RuntimeError(message)
        return result.data
