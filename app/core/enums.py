from __future__ import annotations

from enum import StrEnum


class AgentKind(StrEnum):
    PLANNER = "planner"
    DATA_ANALYST = "data_analyst"
    PARSER = "parser"
    NLP = "nlp"
    KNOWLEDGE = "knowledge"
    REVIEWER = "reviewer"
    REPORT = "report"


class HarnessKind(StrEnum):
    CONTEXT = "context"
    PLANNER = "planner"
    TOOL = "tool"
    EXECUTION = "execution"
    PERMISSION = "permission"
    VALIDATION = "validation"
    MEMORY = "memory"
    OBSERVABILITY = "observability"
    EVALUATION = "evaluation"


class McpCapability(StrEnum):
    DATA_ANALYSIS = "data_analysis"
    FILESYSTEM = "filesystem"
    POSTGRESQL = "postgresql"
    DUCKDB = "duckdb"
    NEO4J = "neo4j"
    VECTOR = "vector"
    NLP = "nlp"
    GITHUB = "github"
    SCHEDULER = "scheduler"
    NOTIFICATION = "notification"
    MODEL_ROUTER = "model_router"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
