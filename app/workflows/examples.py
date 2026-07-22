from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

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
from app.core.entities import ExecutionPlan, PlanStep, TaskIntent
from app.core.enums import AgentKind, McpCapability
from app.knowledge.models import KnowledgeUpdateResult, KnowledgeUpdateStatus
from app.workflows.graph import build_datamind_workflow, initial_workflow_state
from app.workflows.models import (
    CrawlOutput,
    ReportResult,
    ReviewResult,
    SearchOutput,
    WorkflowState,
)
from app.workflows.nodes import WorkflowAgents


@dataclass
class DataAnalysisAgents:
    reviewer_failures_before_pass: int = 0
    planner: DataAnalysisPlannerAgent = field(init=False)
    search: DataProfilingAgent = field(init=False)
    crawl: DataAggregationAgent = field(init=False)
    parser: DataParserAgent = field(init=False)
    nlp: DataInsightAgent = field(init=False)
    knowledge: DataKnowledgeAgent = field(init=False)
    reviewer: DataReviewerAgent = field(init=False)
    report: DataReportAgent = field(init=False)

    def __post_init__(self) -> None:
        self.planner = DataAnalysisPlannerAgent()
        self.search = DataProfilingAgent()
        self.crawl = DataAggregationAgent()
        self.parser = DataParserAgent()
        self.nlp = DataInsightAgent()
        self.knowledge = DataKnowledgeAgent()
        self.reviewer = DataReviewerAgent(self.reviewer_failures_before_pass)
        self.report = DataReportAgent()


class DataAnalysisPlannerAgent:
    async def plan(self, task: TaskIntent) -> ExecutionPlan:
        profile = PlanStep(
            name="Profile dataset",
            agent=AgentKind.DATA_ANALYST,
            description="Inspect schema, missingness, and numeric distributions.",
            required_capabilities=(McpCapability.DATA_ANALYSIS,),
        )
        aggregate = PlanStep(
            name="Aggregate metrics",
            agent=AgentKind.DATA_ANALYST,
            description="Aggregate metrics by business dimensions.",
            required_capabilities=(McpCapability.DATA_ANALYSIS,),
            depends_on=(profile.step_id,),
        )
        parser = PlanStep(
            name="Normalize analysis context",
            agent=AgentKind.PARSER,
            description="Normalize analytical findings for downstream insight extraction.",
            depends_on=(aggregate.step_id,),
        )
        nlp = PlanStep(
            name="Generate insight narrative",
            agent=AgentKind.NLP,
            description="Extract concise insights and natural-language summary.",
            required_capabilities=(McpCapability.NLP, McpCapability.MODEL_ROUTER),
            depends_on=(parser.step_id,),
        )
        knowledge = PlanStep(
            name="Record analysis memory",
            agent=AgentKind.KNOWLEDGE,
            description="Persist analysis metadata, evidence, and reusable insight context.",
            required_capabilities=(
                McpCapability.POSTGRESQL,
                McpCapability.VECTOR,
            ),
            depends_on=(nlp.step_id,),
        )
        review = PlanStep(
            name="Review analysis quality",
            agent=AgentKind.REVIEWER,
            description="Validate metric consistency, anomaly reasoning, and report confidence.",
            depends_on=(knowledge.step_id,),
        )
        report = PlanStep(
            name="Generate analysis report",
            agent=AgentKind.REPORT,
            description="Produce Markdown data analysis report.",
            depends_on=(review.step_id,),
        )
        return ExecutionPlan(
            task=task,
            objective="Analyze regional sales performance and unusual revenue patterns.",
            steps=(profile, aggregate, parser, nlp, knowledge, review, report),
        )


class DataProfilingAgent:
    async def search(self, plan: ExecutionPlan) -> tuple[SearchOutput, ...]:
        return (
            SearchOutput(
                query=plan.objective,
                urls=("https://example.com/datasets/sample-sales/profile",),
            ),
        )


class DataAggregationAgent:
    async def crawl(self, search_results: tuple[SearchOutput, ...]) -> tuple[CrawlOutput, ...]:
        return tuple(
            CrawlOutput(
                url=url,
                content=(
                    "销售分析结果: 华北与华东收入表现稳定, 西部存在一个明显偏高的收入观测值。"
                ),
                content_type="application/json",
            )
            for result in search_results
            for url in result.urls
        )


class DataParserAgent:
    async def parse(self, crawl_results: tuple[CrawlOutput, ...]) -> tuple[ParsedDocument, ...]:
        return tuple(
            ParsedDocument(
                document_id=f"00000000-0000-0000-0000-{index:012d}",
                source_url=result.url,
                title="销售分析上下文",
                normalized_text=result.content,
                content_hash=f"analysis-{index}",
            )
            for index, result in enumerate(crawl_results, start=1)
        )


class DataInsightAgent:
    async def analyze(self, documents: tuple[ParsedDocument, ...]) -> tuple[NLPExtraction, ...]:
        return tuple(
            NLPExtraction(
                document_id=document.document_id,
                source_url=document.source_url,
                backend=NLPBackendKind.RULE_BASED,
                entities=(
                    Entity(text="华北", label="REGION", confidence=0.9),
                    Entity(text="西部", label="REGION", confidence=0.9),
                    Entity(text="收入", label="METRIC", confidence=0.85),
                ),
                relations=(
                    Relation(
                        subject="西部",
                        predicate="has_anomaly",
                        object="收入",
                        confidence=0.75,
                        evidence=document.normalized_text,
                    ),
                ),
                keywords=(Keyword(text="销售", score=0.9), Keyword(text="异常", score=0.8)),
                topics=(Topic(label="销售表现", score=0.85),),
                summary=Summary(text="区域销售整体稳定, 但存在一个收入异常点。"),
                language=LanguageDetection(language="en", confidence=0.95),
                sentiment=SentimentAnalysis(sentiment="neutral", confidence=0.7),
            )
            for document in documents
        )


class DataKnowledgeAgent:
    async def update(
        self,
        documents: tuple[ParsedDocument, ...],
        extractions: tuple[NLPExtraction, ...],
    ) -> tuple[tuple[KnowledgeReadyDocument, ...], KnowledgeUpdateResult]:
        ready = tuple(
            KnowledgeReadyDocument(document=document, extraction=extraction)
            for document, extraction in zip(documents, extractions, strict=True)
        )
        return ready, KnowledgeUpdateResult(
            document_id=documents[0].document_id,
            status=KnowledgeUpdateStatus.UPDATED,
            entities_upserted=sum(len(item.entities) for item in extractions),
            relations_upserted=sum(len(item.relations) for item in extractions),
            vectors_upserted=len(documents),
            timeline_events_upserted=len(documents),
        )


class DataReviewerAgent:
    def __init__(self, failures_before_pass: int = 0) -> None:
        self._failures_before_pass = failures_before_pass
        self.calls = 0

    async def review(self, state_summary: dict[str, object]) -> ReviewResult:
        self.calls += 1
        passed = self.calls > self._failures_before_pass
        return ReviewResult(
            passed=passed,
            confidence=0.88 if passed else 0.4,
            findings=() if passed else ("Aggregation and anomaly evidence need review.",),
        )


class DataReportAgent:
    async def report(self, state_summary: dict[str, object]) -> ReportResult:
        return ReportResult(
            title="销售表现分析",
            markdown=(
                "# 销售表现分析\n\n"
                "- 已通过数据分析 MCP 完成区域收入画像与聚合。\n"
                "- 西部存在一个明显偏高的收入观测值, 需要进一步复核。\n"
                "- 分析元数据已准备好用于持久化和后续报告生成。\n"
            ),
        )


async def run_data_analysis_example() -> WorkflowState:
    task = TaskIntent(
        tenant_id="demo",
        user_id="system",
        prompt="Analyze regional sales performance and identify unusual revenue patterns.",
    )
    return await run_data_analysis_for_task(task)


async def run_data_analysis_for_task(task: TaskIntent) -> WorkflowState:
    agents = DataAnalysisAgents()
    workflow = build_datamind_workflow(cast(WorkflowAgents, agents))
    result = await workflow.ainvoke(initial_workflow_state(task))
    return cast(WorkflowState, result)


DailyMCPMonitoringAgents = DataAnalysisAgents
run_daily_mcp_monitoring_example = run_data_analysis_example
run_daily_mcp_monitoring_for_task = run_data_analysis_for_task
