from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.core.entities import Evidence, ExecutionPlan


class PlanRepository(Protocol):
    async def save(self, plan: ExecutionPlan) -> None:
        """Persist an execution plan."""

    async def get(self, plan_id: UUID) -> ExecutionPlan | None:
        """Retrieve an execution plan."""


class EvidenceRepository(Protocol):
    async def append(self, plan_id: UUID, evidence: Evidence) -> None:
        """Persist evidence collected by MCP-backed capabilities."""

    async def list_by_plan(self, plan_id: UUID) -> tuple[Evidence, ...]:
        """List evidence for a plan."""
