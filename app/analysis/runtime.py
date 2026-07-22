from __future__ import annotations

from typing import Any, Protocol

from app.analysis.model_router import MCPAnalysisModelRouter
from app.analysis.prompt_override_router import PromptOverrideModelRouter
from app.analysis.workflow import AnalysisWorkflowRunner
from app.storage.dataset_store import DatasetStoreRepository


class AnalysisRunnerProvider(Protocol):
    def __call__(self, repository: DatasetStoreRepository) -> AnalysisWorkflowRunner: ...


def build_analysis_runner(
    repository: DatasetStoreRepository,
    *,
    prompt_overrides: dict[str, Any] | None = None,
) -> AnalysisWorkflowRunner:
    router = PromptOverrideModelRouter(MCPAnalysisModelRouter(), prompt_overrides)
    return AnalysisWorkflowRunner(repository, model_router=router)
