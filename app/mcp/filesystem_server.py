from __future__ import annotations

from uuid import UUID

from app.analysis.services import DatasetProfiler
from app.core.enums import McpCapability
from app.mcp.models import MCPServer, MCPTool
from app.storage.dataset_store import DatasetStoreRepository


class FilesystemMCPServer:
    def __init__(
        self,
        repository: DatasetStoreRepository,
        name: str = "filesystem-mcp",
    ) -> None:
        self._repository = repository
        self._server = MCPServer(
            name=name,
            description="Filesystem MCP server for local dataset and report access.",
            tools=(
                _filesystem_tool(
                    "filesystem_upload_dataset",
                    "Create a local dataset and store raw records.",
                    {
                        "type": "object",
                        "required": ["name", "source_type", "records"],
                        "properties": {
                            "name": {"type": "string"},
                            "source_type": {"type": "string"},
                            "source_metadata": {"type": "object"},
                            "records": {"type": "array"},
                        },
                    },
                ),
                _filesystem_tool(
                    "filesystem_list_datasets",
                    "List local datasets.",
                    {"type": "object", "properties": {}},
                ),
                _filesystem_tool(
                    "filesystem_read_preview",
                    "Read raw dataset preview rows.",
                    {
                        "type": "object",
                        "required": ["dataset_id"],
                        "properties": {
                            "dataset_id": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                ),
                _filesystem_tool(
                    "filesystem_profile_dataset",
                    "Profile a stored local dataset.",
                    {
                        "type": "object",
                        "required": ["dataset_id"],
                        "properties": {"dataset_id": {"type": "string"}},
                    },
                ),
                _filesystem_tool(
                    "filesystem_save_report",
                    "Save a Markdown report for a local dataset.",
                    {
                        "type": "object",
                        "required": ["dataset_id", "title", "markdown"],
                        "properties": {
                            "dataset_id": {"type": "string"},
                            "title": {"type": "string"},
                            "markdown": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                    },
                ),
            ),
        )

    @property
    def server(self) -> MCPServer:
        return self._server

    async def list_tools(self) -> tuple[MCPTool, ...]:
        return self._server.tools

    async def invoke(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        match tool_name:
            case "filesystem_upload_dataset":
                return self._upload_dataset(arguments)
            case "filesystem_list_datasets":
                return self._list_datasets()
            case "filesystem_read_preview":
                return self._read_preview(arguments)
            case "filesystem_profile_dataset":
                return self._profile_dataset(arguments)
            case "filesystem_save_report":
                return self._save_report(arguments)
            case _:
                raise LookupError(f"Unknown Filesystem MCP tool: {tool_name}")

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        return await self.invoke(tool_name, arguments)

    def _upload_dataset(self, arguments: dict[str, object]) -> dict[str, object]:
        records = arguments.get("records")
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError("records must be a list of objects.")
        source_metadata = arguments.get("source_metadata")
        dataset = self._repository.create_dataset(
            name=str(arguments["name"]),
            source_type=str(arguments["source_type"]),
            source_metadata=source_metadata if isinstance(source_metadata, dict) else {},
        )
        inserted = self._repository.append_raw_records(
            dataset_id=dataset.id,
            records=[dict(row) for row in records],
        )
        return {
            "dataset_id": str(dataset.id),
            "name": dataset.name,
            "source_type": dataset.source_type,
            "inserted": inserted,
        }

    def _list_datasets(self) -> dict[str, object]:
        return {
            "datasets": [
                {
                    "dataset_id": str(dataset.id),
                    "name": dataset.name,
                    "source_type": dataset.source_type,
                    "status": dataset.status,
                    "created_at": dataset.created_at,
                    "updated_at": dataset.updated_at,
                }
                for dataset in self._repository.list_datasets()
            ]
        }

    def _read_preview(self, arguments: dict[str, object]) -> dict[str, object]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        limit = int(arguments.get("limit") or 20)
        return {
            "dataset_id": str(dataset_id),
            "records": self._repository.preview_raw_records(dataset_id, limit=limit),
        }

    def _profile_dataset(self, arguments: dict[str, object]) -> dict[str, object]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        records = self._repository.read_raw_records(dataset_id)
        profile = DatasetProfiler().profile(dataset_id=dataset_id, records=records)
        return profile.model_dump(mode="json")

    def _save_report(self, arguments: dict[str, object]) -> dict[str, object]:
        metadata = arguments.get("metadata")
        report_id = self._repository.save_report(
            dataset_id=UUID(str(arguments["dataset_id"])),
            title=str(arguments["title"]),
            markdown=str(arguments["markdown"]),
            metadata=metadata if isinstance(metadata, dict) else {},
        )
        return {"report_id": str(report_id)}


def _filesystem_tool(name: str, description: str, input_schema: dict[str, object]) -> MCPTool:
    return MCPTool(
        name=name,
        capability=McpCapability.FILESYSTEM,
        description=description,
        input_schema=input_schema,
        output_schema={"type": "object"},
        timeout_seconds=30.0,
        max_retries=0,
    )
