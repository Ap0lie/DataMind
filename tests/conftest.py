from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.core.settings import get_settings
from tests.fakes import ScriptedPythonExecutor

_CATEGORY_MARKERS = ("unit", "workflow", "integration", "sandbox", "benchmark", "e2e")
_WORKFLOW_MODULES = {"test_analysis_workflow.py", "test_langgraph_workflow.py"}
_SANDBOX_MODULES = {"test_python_sandbox.py"}
_INTEGRATION_MODULES = {
    "test_analysis_service.py",
    "test_api_analysis.py",
    "test_api_mcp.py",
    "test_api_tasks.py",
    "test_data_cleaning.py",
    "test_dataset_group_performance.py",
    "test_multidataset_rules.py",
    "test_reliability_foundation.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    errors: list[str] = []
    for item in items:
        categories = [name for name in _CATEGORY_MARKERS if item.get_closest_marker(name)]
        if not categories:
            item.add_marker(getattr(pytest.mark, _category_for_path(Path(str(item.path)))))
            categories = [name for name in _CATEGORY_MARKERS if item.get_closest_marker(name)]
        if len(categories) != 1:
            errors.append(f"{item.nodeid}: expected exactly one test category, found {categories}")
    if errors:
        raise pytest.UsageError("\n".join(errors))


def _category_for_path(path: Path) -> str:
    if "benchmark" in path.parts:
        return "benchmark"
    if "integration" in path.parts:
        return "integration"
    if path.name in _SANDBOX_MODULES:
        return "sandbox"
    if path.name in _WORKFLOW_MODULES:
        return "workflow"
    if path.name in _INTEGRATION_MODULES:
        return "integration"
    return "unit"


@pytest.fixture
def scripted_python_executor() -> ScriptedPythonExecutor:
    return ScriptedPythonExecutor()


@pytest.fixture
def patch_workflow_python_executor(
    monkeypatch: pytest.MonkeyPatch,
    scripted_python_executor: ScriptedPythonExecutor,
) -> ScriptedPythonExecutor:
    monkeypatch.setattr(
        "app.analysis.workflow.run_generated_python_analysis",
        scripted_python_executor,
    )
    return scripted_python_executor


@pytest.fixture(autouse=True)
def fast_python_executor_for_core_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (
        request.node.get_closest_marker("workflow")
        or request.node.get_closest_marker("integration")
    ):
        return
    monkeypatch.setattr(
        "app.analysis.workflow.run_generated_python_analysis",
        ScriptedPythonExecutor(),
    )


@pytest.fixture(autouse=True)
def force_mock_providers_for_core_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    if not (
        request.node.get_closest_marker("workflow")
        or request.node.get_closest_marker("integration")
    ):
        yield
        return
    for name in (
        "DATAMIND_DEFAULT_LLM_PROVIDER",
        "DATAMIND_CLEANING_LLM_PROVIDER",
        "DATAMIND_PLANNER_LLM_PROVIDER",
        "DATAMIND_SQL_LLM_PROVIDER",
        "DATAMIND_PYTHON_LLM_PROVIDER",
        "DATAMIND_REFLECTION_LLM_PROVIDER",
        "DATAMIND_REPORT_LLM_PROVIDER",
        "DATAMIND_REVIEW_LLM_PROVIDER",
        "DATAMIND_MULTIMODAL_LLM_PROVIDER",
        "DATAMIND_AGENT_LOOP_PROVIDER",
        "DATAMIND_ASSISTANT_LLM_PROVIDER",
    ):
        monkeypatch.setenv(name, "mock")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def forbid_subprocess_in_unit_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not request.node.get_closest_marker("unit"):
        return

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unit tests must not start subprocesses")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
