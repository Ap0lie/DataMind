from __future__ import annotations


def test_architecture_modules_are_importable() -> None:
    import app.agents.contracts
    import app.agents.document_models
    import app.agents.nlp_agent
    import app.agents.parser_agent
    import app.analysis.agent_prompts
    import app.analysis.workflow_nodes
    import app.analysis.workflow_prompts
    import app.analysis.workflow_report_nodes
    import app.analysis.workflow_support
    import app.api.router
    import app.api.v1.mcp
    import app.core.entities
    import app.crawler.contracts
    import app.crawler.engine
    import app.crawler.models
    import app.crawler.strategies
    import app.evaluation.models
    import app.evaluation.services
    import app.harness.contracts
    import app.harness.models
    import app.harness.runtime
    import app.knowledge.agent
    import app.knowledge.contracts
    import app.knowledge.memory
    import app.knowledge.models
    import app.knowledge.services
    import app.mcp.bootstrap
    import app.mcp.contracts
    import app.mcp.data_analysis_server
    import app.mcp.mock_server
    import app.mcp.model_router_server
    import app.mcp.models
    import app.mcp.nlp_server
    import app.mcp.runtime
    import app.mcp.tool_schemas
    import app.schemas.mcp
    import app.schemas.semantic
    import app.schemas.tasks
    import app.semantic.dsl
    import app.semantic.service
    import app.storage.data_reliability_repository
    import app.storage.dataset_group_repository
    import app.storage.dataset_repository
    import app.storage.job_repository
    import app.storage.recycle_repository
    import app.storage.report_repository
    import app.storage.repositories
    import app.storage.repository_utils
    import app.storage.semantic_repository
    import app.workflows.examples
    import app.workflows.graph
    import app.workflows.models
    import app.workflows.nodes

    assert app.api.router.api_router is not None
