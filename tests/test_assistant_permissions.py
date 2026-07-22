from __future__ import annotations

from pathlib import Path

import pytest

from app.assistant.permissions import AssistantPermissionService
from app.core.settings import get_settings
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository


def test_ask_mode_never_authorizes_write_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "datasets"))
    get_settings.cache_clear()
    store = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="alice")
    assistant = AssistantRepository(str(tmp_path / "datasets"), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    assistant.save_permission_grant(
        asset_type="dataset", asset_id=dataset.id, capabilities=("analysis_manage",)
    )
    service = AssistantPermissionService(store=store, assistant_store=assistant)

    with pytest.raises(PermissionError, match="ask mode"):
        service.authorize_tool(
            tool_name="start_analysis",
            arguments={"dataset_id": str(dataset.id)},
            conversation={"scope_type": "auto", "scope_id": None},
            execution_mode="ask",
        )
    get_settings.cache_clear()


def test_group_grant_inherits_to_recycled_member_and_revokes_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "datasets"))
    get_settings.cache_clear()
    store = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="alice")
    assistant = AssistantRepository(str(tmp_path / "datasets"), user_id="alice")
    dataset = store.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    group = store.create_dataset_group(name="commerce", dataset_ids=(dataset.id,))
    grant = assistant.save_permission_grant(
        asset_type="dataset_group",
        asset_id=group.id,
        capabilities=("analysis_manage", "asset_recycle"),
    )
    service = AssistantPermissionService(store=store, assistant_store=assistant)
    conversation = {"scope_type": "dataset_group", "scope_id": group.id}

    authorized = service.authorize_tool(
        tool_name="start_analysis",
        arguments={"dataset_id": str(dataset.id)},
        conversation=conversation,
        execution_mode="execute",
    )
    assert authorized is not None and authorized.grant_id == grant["grant_id"]

    store.soft_delete_asset(asset_type="dataset", asset_id=dataset.id)
    restored = service.authorize_tool(
        tool_name="restore_asset",
        arguments={"asset_type": "dataset", "asset_id": str(dataset.id)},
        conversation=conversation,
        execution_mode="execute",
    )
    assert restored is not None

    assistant.revoke_permission_grant(grant["grant_id"])
    with pytest.raises(PermissionError, match="Missing assistant capability"):
        service.authorize_tool(
            tool_name="restore_asset",
            arguments={"asset_type": "dataset", "asset_id": str(dataset.id)},
            conversation=conversation,
            execution_mode="execute",
        )
    get_settings.cache_clear()


def test_report_grant_can_analyze_only_its_source_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "datasets"))
    get_settings.cache_clear()
    store = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="alice")
    assistant = AssistantRepository(str(tmp_path / "datasets"), user_id="alice")
    source = store.create_dataset(name="source.csv", source_type="csv", source_metadata={})
    other = store.create_dataset(name="other.csv", source_type="csv", source_metadata={})
    report_id = store.save_report(
        dataset_id=source.id,
        title="Source report",
        markdown="verified",
        metadata={"question": "source question"},
    )
    assistant.save_permission_grant(
        asset_type="report",
        asset_id=report_id,
        capabilities=("analysis_manage", "report_manage"),
    )
    service = AssistantPermissionService(store=store, assistant_store=assistant)
    conversation = {"scope_type": "report", "scope_id": report_id}

    authorized = service.authorize_tool(
        tool_name="start_analysis",
        arguments={"dataset_id": str(source.id)},
        conversation=conversation,
        execution_mode="execute",
    )
    assert authorized is not None
    revision = service.authorize_tool(
        tool_name="revise_report",
        arguments={"report_id": str(report_id), "instruction": "精简报告并美化图表"},
        conversation=conversation,
        execution_mode="execute",
    )
    assert revision is not None and revision.asset_id == report_id

    with pytest.raises(PermissionError, match="outside the conversation scope"):
        service.authorize_tool(
            tool_name="start_analysis",
            arguments={"dataset_id": str(other.id)},
            conversation=conversation,
            execution_mode="execute",
        )
    get_settings.cache_clear()
