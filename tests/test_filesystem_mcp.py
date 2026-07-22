from __future__ import annotations

import pytest

from app.core.enums import McpCapability
from app.mcp.filesystem_server import FilesystemMCPServer
from app.mcp.runtime import InMemoryMCPRuntime
from app.storage.dataset_store import DatasetStoreRepository


@pytest.mark.asyncio
async def test_filesystem_mcp_uploads_reads_profiles_and_saves_report(tmp_path) -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(FilesystemMCPServer(DatasetStoreRepository(str(tmp_path))))

    registry = await runtime.discover(McpCapability.FILESYSTEM)
    assert {tool.name for tool in registry.tools} == {
        "filesystem_upload_dataset",
        "filesystem_list_datasets",
        "filesystem_read_preview",
        "filesystem_profile_dataset",
        "filesystem_save_report",
    }

    upload = await runtime.invoke(
        "filesystem_upload_dataset",
        {
            "name": "sales.csv",
            "source_type": "csv",
            "records": [
                {"region": "East", "sales": 10},
                {"region": "West", "sales": 30},
            ],
        },
    )
    dataset_id = upload.data["dataset_id"]
    preview = await runtime.invoke(
        "filesystem_read_preview",
        {"dataset_id": dataset_id, "limit": 1},
    )
    profile = await runtime.invoke("filesystem_profile_dataset", {"dataset_id": dataset_id})
    report = await runtime.invoke(
        "filesystem_save_report",
        {
            "dataset_id": dataset_id,
            "title": "DataMind 分析报告",
            "markdown": "# DataMind 分析报告",
            "metadata": {"source": "test"},
        },
    )

    assert upload.ok
    assert upload.data["inserted"] == 2
    assert preview.ok
    assert preview.data["records"] == [{"region": "East", "sales": 10}]
    assert profile.ok
    assert profile.data["row_count"] == 2
    assert report.ok
    assert report.data["report_id"]
