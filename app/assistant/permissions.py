from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository

FULL_CAPABILITIES: tuple[str, ...] = (
    "data_prepare",
    "relationship_manage",
    "analysis_manage",
    "report_manage",
    "semantic_manage",
    "asset_recycle",
)

ASSET_CAPABILITY_MATRIX: dict[str, tuple[str, ...]] = {
    "dataset": (
        "data_prepare",
        "analysis_manage",
        "report_manage",
        "semantic_manage",
        "asset_recycle",
    ),
    "dataset_group": FULL_CAPABILITIES,
    "report": (
        "analysis_manage",
        "report_manage",
        "asset_recycle",
    ),
    "semantic_model": (
        "semantic_manage",
        "asset_recycle",
    ),
}


def capabilities_for_asset(asset_type: str) -> tuple[str, ...]:
    try:
        return ASSET_CAPABILITY_MATRIX[asset_type]
    except KeyError as exc:
        raise ValueError("Unsupported assistant grant asset type.") from exc

WRITE_TOOL_CAPABILITY: dict[str, str] = {
    "start_analysis": "analysis_manage",
    "start_cleaning": "data_prepare",
    "activate_cleaning_version": "data_prepare",
    "rollback_cleaning_version": "data_prepare",
    "update_column_metadata": "data_prepare",
    "save_relationship_plan": "relationship_manage",
    "cancel_analysis": "analysis_manage",
    "retry_analysis": "analysis_manage",
    "rename_report": "report_manage",
    "revise_report": "report_manage",
    "create_semantic_draft": "semantic_manage",
    "update_semantic_draft": "semantic_manage",
    "publish_semantic_model": "semantic_manage",
    "soft_delete_asset": "asset_recycle",
    "restore_asset": "asset_recycle",
}


@dataclass(frozen=True)
class AuthorizedAsset:
    asset_type: str
    asset_id: UUID
    capability: str
    grant_id: UUID


class AssistantPermissionService:
    """Server-side capability checks for Kimi write tools."""

    def __init__(
        self, *, store: DatasetStoreRepository, assistant_store: AssistantRepository
    ) -> None:
        self.store = store
        self.assistant_store = assistant_store

    def validate_grant_target(self, asset_type: str, asset_id: UUID) -> None:
        if asset_type == "dataset":
            self.store.get_dataset(asset_id)
        elif asset_type == "dataset_group":
            self.store.get_dataset_group(asset_id)
        elif asset_type == "report":
            self.store.get_report(asset_id)
        elif asset_type == "semantic_model":
            self.store.get_semantic_model(asset_id)
        else:
            raise ValueError("Unsupported assistant grant asset type.")

    def validate_grant_capabilities(
        self,
        asset_type: str,
        capabilities: tuple[str, ...],
    ) -> None:
        allowed = set(capabilities_for_asset(asset_type))
        unsupported = sorted(set(capabilities) - allowed)
        if unsupported:
            raise ValueError(
                f"Capabilities are not allowed for {asset_type}: {', '.join(unsupported)}."
            )

    def authorize_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        conversation: dict[str, Any],
        execution_mode: str,
    ) -> AuthorizedAsset | None:
        capability = WRITE_TOOL_CAPABILITY.get(tool_name)
        if capability is None:
            return None
        if execution_mode != "execute":
            raise PermissionError("Write tools are unavailable in ask mode.")
        asset_type, asset_id = self._tool_asset(tool_name, arguments)
        if not self._scope_allows(conversation, asset_type, asset_id):
            raise PermissionError("Asset is outside the conversation scope.")
        grant = self._matching_grant(asset_type, asset_id, capability)
        if grant is None:
            raise PermissionError(f"Missing assistant capability: {capability}.")
        return AuthorizedAsset(
            asset_type=asset_type,
            asset_id=asset_id,
            capability=capability,
            grant_id=grant["grant_id"],
        )

    def _tool_asset(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, UUID]:
        if tool_name == "start_analysis":
            return "dataset", UUID(str(arguments["dataset_id"]))
        if tool_name in {
            "start_cleaning",
            "activate_cleaning_version",
            "rollback_cleaning_version",
            "update_column_metadata",
        }:
            return "dataset", UUID(str(arguments["dataset_id"]))
        if tool_name == "save_relationship_plan":
            return "dataset_group", UUID(str(arguments["dataset_group_id"]))
        if tool_name in {"cancel_analysis", "retry_analysis"}:
            job = self.store.get_analysis_job(UUID(str(arguments["job_id"])))
            return "dataset", job.dataset_id
        if tool_name in {"rename_report", "revise_report"}:
            return "report", UUID(str(arguments["report_id"]))
        if tool_name == "create_semantic_draft":
            scope_type = str(arguments["scope_type"])
            return scope_type, UUID(str(arguments["scope_id"]))
        if tool_name in {"update_semantic_draft", "publish_semantic_model"}:
            return "semantic_model", UUID(str(arguments["model_id"]))
        if tool_name in {"soft_delete_asset", "restore_asset"}:
            return str(arguments["asset_type"]), UUID(str(arguments["asset_id"]))
        raise PermissionError("Write tool has no server-side asset mapping.")

    def _matching_grant(
        self, asset_type: str, asset_id: UUID, capability: str
    ) -> dict[str, Any] | None:
        grants = self.assistant_store.list_permission_grants()
        for grant in grants:
            if capability not in capabilities_for_asset(grant["asset_type"]):
                continue
            if capability not in grant["capabilities"]:
                continue
            if grant["asset_type"] == asset_type and grant["asset_id"] == asset_id:
                return grant
            if self._grant_inherits(grant, asset_type, asset_id, capability):
                return grant
        return None

    def _grant_inherits(
        self,
        grant: dict[str, Any],
        asset_type: str,
        asset_id: UUID,
        capability: str,
    ) -> bool:
        try:
            if asset_type == "report":
                report = self.store.get_report(asset_id)
                return self._grant_covers_dataset(
                    grant,
                    UUID(str(report["dataset_id"])),
                    capability,
                )
            if asset_type == "semantic_model":
                model = self.store.get_semantic_model(asset_id)
                return grant["asset_type"] == model["scope_type"] and grant["asset_id"] == UUID(
                    str(model["scope_id"])
                )
            if asset_type == "dataset":
                return self._grant_covers_dataset(grant, asset_id, capability)
        except RuntimeError:
            return grant["asset_type"] == asset_type and grant["asset_id"] == asset_id
        return False

    def _grant_covers_dataset(
        self,
        grant: dict[str, Any],
        dataset_id: UUID,
        capability: str,
    ) -> bool:
        if grant["asset_type"] == "dataset":
            return grant["asset_id"] == dataset_id
        if grant["asset_type"] == "report":
            if capability != "analysis_manage":
                return False
            try:
                report = self.store.get_report(grant["asset_id"])
            except RuntimeError:
                return False
            return UUID(str(report["dataset_id"])) == dataset_id
        if grant["asset_type"] == "dataset_group":
            return self.store.dataset_group_contains_dataset(
                group_id=grant["asset_id"],
                dataset_id=dataset_id,
                include_recycled=True,
            )
        return False

    def _scope_allows(self, conversation: dict[str, Any], asset_type: str, asset_id: UUID) -> bool:
        scope_type = str(conversation["scope_type"])
        scope_id = conversation.get("scope_id")
        if scope_type == "auto":
            return True
        if scope_id is None:
            return False
        scope_id = UUID(str(scope_id))
        if scope_type == asset_type and scope_id == asset_id:
            return True
        try:
            if asset_type == "report":
                dataset_id = UUID(str(self.store.get_report(asset_id)["dataset_id"]))
                return self._scope_dataset(scope_type, scope_id, dataset_id)
            if asset_type == "semantic_model":
                model = self.store.get_semantic_model(asset_id)
                return (
                    str(model["scope_type"]) == scope_type
                    and UUID(str(model["scope_id"])) == scope_id
                )
            if asset_type == "dataset":
                return self._scope_dataset(scope_type, scope_id, asset_id)
        except RuntimeError:
            return scope_type == asset_type and scope_id == asset_id
        return False

    def _scope_dataset(self, scope_type: str, scope_id: UUID, dataset_id: UUID) -> bool:
        if scope_type == "dataset":
            return scope_id == dataset_id
        if scope_type == "dataset_group":
            return self.store.dataset_group_contains_dataset(
                group_id=scope_id,
                dataset_id=dataset_id,
                include_recycled=True,
            )
        if scope_type == "report":
            return UUID(str(self.store.get_report(scope_id)["dataset_id"])) == dataset_id
        return False
