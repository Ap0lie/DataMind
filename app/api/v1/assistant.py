from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from app.analysis.cleaning_jobs import start_cleaning_job
from app.analysis.dataset_groups import (
    select_automatic_dataset_relationships,
    suggest_dataset_group_relationships,
)
from app.api.v1.deps import current_user_id
from app.assistant.evidence import canonical_reliability, safe_excerpt
from app.assistant.jobs import start_assistant_run
from app.assistant.memory import (
    MEMORY_SCOPES,
    MEMORY_STATUSES,
    MEMORY_TYPES,
    AssistantMemoryService,
)
from app.assistant.permissions import (
    AssistantPermissionService,
    capabilities_for_asset,
)
from app.core.settings import get_settings
from app.schemas.assistant import (
    AssistantActionListResponse,
    AssistantActionResponse,
    AssistantAttachmentResponse,
    AssistantConversationCreateRequest,
    AssistantConversationListResponse,
    AssistantConversationResponse,
    AssistantConversationUpdateRequest,
    AssistantImportBatchCommitRequest,
    AssistantImportBatchPreviewRequest,
    AssistantImportBatchResponse,
    AssistantMemoryCreateRequest,
    AssistantMemoryEffectivenessResponse,
    AssistantMemoryFeedbackRequest,
    AssistantMemoryFeedbackResponse,
    AssistantMemoryHistoryResponse,
    AssistantMemoryListResponse,
    AssistantMemoryResponse,
    AssistantMemorySettingsResponse,
    AssistantMemorySettingsUpdateRequest,
    AssistantMemoryUpdateRequest,
    AssistantMemoryUsageListResponse,
    AssistantMemoryUsageResponse,
    AssistantMessageCreateRequest,
    AssistantMessageListResponse,
    AssistantMessageResponse,
    AssistantPermissionGrantCreateRequest,
    AssistantPermissionGrantListResponse,
    AssistantPermissionGrantResponse,
    AssistantRunConfirmRequest,
    AssistantRunResponse,
    RecycledAssetListResponse,
    RecycledAssetResponse,
)
from app.security.rate_limit import RateLimitExceeded, enforce_rate_limit
from app.services.tabular_import import (
    preview_file_from_path,
    record_batches_from_file_path,
    xlsx_sheet_previews_from_path,
)
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.assistant_repository import AssistantRepository, StoredAssistantRun
from app.storage.dataset_store import DatasetStoreRepository

router = APIRouter()


def _repository(user_id: str) -> AssistantRepository:
    return AssistantRepository(get_settings().dataset_store_path, user_id=user_id)


def _memory_repository(user_id: str) -> AssistantMemoryRepository:
    return AssistantMemoryRepository(get_settings().dataset_store_path, user_id=user_id)


def _memory_service(user_id: str) -> AssistantMemoryService:
    settings = get_settings()
    return AssistantMemoryService(
        repository=_memory_repository(user_id),
        store=DatasetStoreRepository(settings.dataset_store_path, user_id=user_id),
        settings=settings,
    )


def _action_response(action: dict[str, Any]) -> AssistantActionResponse:
    public_fields = AssistantActionResponse.model_fields
    return AssistantActionResponse.model_validate(
        {field: action.get(field) for field in public_fields}
    )


def _memory_response(memory: dict[str, Any]) -> AssistantMemoryResponse:
    public_fields = AssistantMemoryResponse.model_fields
    return AssistantMemoryResponse.model_validate(
        {field: memory.get(field) for field in public_fields}
    )


@router.post("/conversations", response_model=AssistantConversationResponse, status_code=201)
def create_conversation(
    request: AssistantConversationCreateRequest, user_id: str = Depends(current_user_id)
) -> AssistantConversationResponse:
    settings = get_settings()
    if not settings.assistant_enabled:
        raise HTTPException(status_code=503, detail="Kimi assistant is disabled.")
    try:
        return AssistantConversationResponse.model_validate(
            _repository(user_id).create_conversation(
                title=request.title or "新对话",
                scope_type=request.scope_type,
                scope_id=request.scope_id,
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/conversations", response_model=AssistantConversationListResponse)
def list_conversations(
    user_id: str = Depends(current_user_id),
) -> AssistantConversationListResponse:
    return AssistantConversationListResponse(
        conversations=tuple(
            AssistantConversationResponse.model_validate(item)
            for item in _repository(user_id).list_conversations()
        )
    )


@router.get("/conversations/{conversation_id}", response_model=AssistantConversationResponse)
def get_conversation(
    conversation_id: UUID, user_id: str = Depends(current_user_id)
) -> AssistantConversationResponse:
    try:
        return AssistantConversationResponse.model_validate(
            _repository(user_id).get_conversation(conversation_id)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/conversations/{conversation_id}", response_model=AssistantConversationResponse)
def update_conversation(
    conversation_id: UUID,
    request: AssistantConversationUpdateRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantConversationResponse:
    try:
        return AssistantConversationResponse.model_validate(
            _repository(user_id).update_conversation(
                conversation_id,
                title=request.title,
                scope_type=request.scope_type,
                scope_id=request.scope_id,
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: UUID, user_id: str = Depends(current_user_id)) -> None:
    try:
        _repository(user_id).delete_conversation(conversation_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/permission-grants", response_model=AssistantPermissionGrantListResponse)
def list_permission_grants(
    user_id: str = Depends(current_user_id),
) -> AssistantPermissionGrantListResponse:
    return AssistantPermissionGrantListResponse(
        grants=tuple(
            AssistantPermissionGrantResponse.model_validate(item)
            for item in _repository(user_id).list_permission_grants()
        )
    )


@router.post("/permission-grants", response_model=AssistantPermissionGrantResponse, status_code=201)
def create_permission_grant(
    request: AssistantPermissionGrantCreateRequest, user_id: str = Depends(current_user_id)
) -> AssistantPermissionGrantResponse:
    repository = _repository(user_id)
    store = DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)
    try:
        permission_service = AssistantPermissionService(
            store=store,
            assistant_store=repository,
        )
        permission_service.validate_grant_target(request.asset_type, request.asset_id)
        if not request.capabilities:
            raise ValueError("At least one assistant capability is required.")
        permission_service.validate_grant_capabilities(
            request.asset_type,
            request.capabilities,
        )
        return AssistantPermissionGrantResponse.model_validate(
            repository.save_permission_grant(
                asset_type=request.asset_type,
                asset_id=request.asset_id,
                capabilities=request.capabilities,
            )
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/permission-grants/{grant_id}", response_model=AssistantPermissionGrantResponse)
def revoke_permission_grant(
    grant_id: UUID, user_id: str = Depends(current_user_id)
) -> AssistantPermissionGrantResponse:
    try:
        return AssistantPermissionGrantResponse.model_validate(
            _repository(user_id).revoke_permission_grant(grant_id)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/memories", response_model=AssistantMemoryListResponse)
def list_memories(
    scope_type: str | None = Query(default=None),
    scope_id: UUID | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    memory_kind: str | None = Query(default=None),
    status: str | None = Query(default=None),
    query: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=200, ge=1, le=500),
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryListResponse:
    if scope_type is not None and scope_type not in MEMORY_SCOPES:
        raise HTTPException(status_code=422, detail="Unsupported assistant memory scope.")
    if memory_type is not None and memory_type not in MEMORY_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported assistant memory type.")
    if memory_kind is not None and memory_kind not in {"semantic", "episodic"}:
        raise HTTPException(status_code=422, detail="Unsupported assistant memory kind.")
    if status is not None and status not in MEMORY_STATUSES:
        raise HTTPException(status_code=422, detail="Unsupported assistant memory status.")
    items = _memory_repository(user_id).list(
        scope_type=scope_type,
        scope_id=scope_id,
        memory_type=memory_type,
        memory_kind=memory_kind,
        status=status,
        query=query,
        limit=limit,
    )
    return AssistantMemoryListResponse(
        memories=tuple(_memory_response(item) for item in items)
    )


@router.post("/memories", response_model=AssistantMemoryResponse, status_code=201)
def create_memory(
    request: AssistantMemoryCreateRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    try:
        item = _memory_service(user_id).create_manual(
            memory_type=request.memory_type,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            content=request.content,
            pinned=request.pinned,
        )
        return _memory_response(item)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/memories/{memory_id}", response_model=AssistantMemoryResponse)
def update_memory(
    memory_id: UUID,
    request: AssistantMemoryUpdateRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    try:
        item = _memory_service(user_id).update_memory(
            memory_id,
            memory_type=request.memory_type,
            content=request.content,
            pinned=request.pinned,
        )
        return _memory_response(item)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/memories/{memory_id}/confirm", response_model=AssistantMemoryResponse)
def confirm_memory(
    memory_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    repository = _memory_repository(user_id)
    try:
        current = repository.get(memory_id)
        if current["status"] != "pending":
            raise ValueError("Only pending memories can be confirmed.")
        return _memory_response(repository.confirm(memory_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/memories/{memory_id}", response_model=AssistantMemoryResponse)
def recycle_memory(
    memory_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    settings = get_settings()
    try:
        return _memory_response(
            _memory_repository(user_id).recycle(
                memory_id,
                retention_days=settings.assistant_memory_recycle_days,
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/memories/{memory_id}/restore", response_model=AssistantMemoryResponse)
def restore_memory(
    memory_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    try:
        return _memory_response(_memory_repository(user_id).restore(memory_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/memory-settings", response_model=AssistantMemorySettingsResponse)
def get_memory_settings(
    user_id: str = Depends(current_user_id),
) -> AssistantMemorySettingsResponse:
    return AssistantMemorySettingsResponse.model_validate(
        _memory_repository(user_id).get_settings()
    )


@router.patch("/memory-settings", response_model=AssistantMemorySettingsResponse)
def update_memory_settings(
    request: AssistantMemorySettingsUpdateRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantMemorySettingsResponse:
    return AssistantMemorySettingsResponse.model_validate(
        _memory_repository(user_id).update_settings(enabled=request.enabled)
    )


@router.get(
    "/memories/{memory_id}/history",
    response_model=AssistantMemoryHistoryResponse,
)
def memory_history(
    memory_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryHistoryResponse:
    try:
        items = _memory_repository(user_id).history(memory_id)
        return AssistantMemoryHistoryResponse(
            subject_key=items[0]["subject_key"],
            memories=tuple(_memory_response(item) for item in items),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/memories/{memory_id}/reactivate",
    response_model=AssistantMemoryResponse,
)
def reactivate_memory(
    memory_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    try:
        return _memory_response(_memory_repository(user_id).reactivate(memory_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/memory-usage", response_model=AssistantMemoryUsageListResponse)
def list_memory_usage(
    run_id: UUID,
    include_suppressed: bool = Query(default=False),
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryUsageListResponse:
    return AssistantMemoryUsageListResponse(
        usages=tuple(
            AssistantMemoryUsageResponse.model_validate(item)
            for item in _memory_repository(user_id).list_usage(
                run_id=run_id,
                include_suppressed=include_suppressed,
            )
        )
    )


@router.post(
    "/memory-usage/{usage_id}/feedback",
    response_model=AssistantMemoryFeedbackResponse,
)
def submit_memory_feedback(
    usage_id: UUID,
    request: AssistantMemoryFeedbackRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryFeedbackResponse:
    settings = get_settings()
    try:
        feedback = _memory_repository(user_id).record_feedback(
            usage_id=usage_id,
            feedback=request.feedback,
            reason=request.reason,
            auto_dormancy=settings.assistant_memory_auto_dormancy_enabled,
            dormancy_threshold=settings.assistant_memory_dormancy_threshold,
            dormancy_min_feedback=settings.assistant_memory_dormancy_min_feedback,
            wrong_feedback_limit=settings.assistant_memory_wrong_feedback_limit,
        )
        return AssistantMemoryFeedbackResponse.model_validate(feedback)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/memory-effectiveness",
    response_model=AssistantMemoryEffectivenessResponse,
)
def memory_effectiveness(
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryEffectivenessResponse:
    settings = get_settings()
    return AssistantMemoryEffectivenessResponse.model_validate(
        _memory_repository(user_id).effectiveness(
            shadow_mode=not settings.assistant_memory_auto_dormancy_enabled,
        )
    )


@router.post("/memories/{memory_id}/wake", response_model=AssistantMemoryResponse)
def wake_memory(
    memory_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AssistantMemoryResponse:
    try:
        return _memory_response(_memory_repository(user_id).wake(memory_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/actions", response_model=AssistantActionListResponse)
def list_actions(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    user_id: str = Depends(current_user_id),
) -> AssistantActionListResponse:
    return AssistantActionListResponse(
        actions=tuple(
            _action_response(item)
            for item in _repository(user_id).list_actions(limit=limit)
        )
    )


@router.post("/actions/{action_id}/undo", response_model=AssistantActionResponse)
def undo_action(
    action_id: UUID, user_id: str = Depends(current_user_id)
) -> AssistantActionResponse:
    repository = _repository(user_id)
    store = DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)
    try:
        action = repository.get_action(action_id)
        if action["status"] != "completed" or not action["reversible"] or action["undone_at"]:
            raise ValueError("该操作不可撤销或已经撤销。")
        _undo_action(store, action)
        return _action_response(repository.mark_action_undone(action_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/recycle-bin", response_model=RecycledAssetListResponse)
def list_recycle_bin(user_id: str = Depends(current_user_id)) -> RecycledAssetListResponse:
    store = DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)
    store.purge_expired_assets()
    return RecycledAssetListResponse(
        assets=tuple(
            RecycledAssetResponse.model_validate(item) for item in store.list_recycled_assets()
        )
    )


@router.post(
    "/recycle-bin/{asset_type}/{asset_id}/restore",
    response_model=RecycledAssetResponse,
)
def restore_recycled_asset(
    asset_type: str,
    asset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> RecycledAssetResponse:
    store = DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)
    try:
        recycled = next(
            item
            for item in store.list_recycled_assets()
            if item["asset_type"] == asset_type and item["asset_id"] == asset_id
        )
        store.restore_asset(asset_type=asset_type, asset_id=asset_id)
        return RecycledAssetResponse.model_validate(recycled)
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail="回收站中没有该资产。") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/conversations/{conversation_id}/messages", response_model=AssistantMessageListResponse
)
def list_messages(
    conversation_id: UUID,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    user_id: str = Depends(current_user_id),
) -> AssistantMessageListResponse:
    repository = _repository(user_id)
    try:
        messages = tuple(
            _message_response(repository, item)
            for item in repository.list_messages(conversation_id, limit=limit)
            if item["role"] != "tool"
        )
        return AssistantMessageListResponse(messages=messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/conversations/{conversation_id}/attachments",
    response_model=AssistantAttachmentResponse,
    status_code=201,
)
async def upload_attachment(
    conversation_id: UUID,
    file: Annotated[UploadFile, File(...)],
    user_id: str = Depends(current_user_id),
) -> AssistantAttachmentResponse:
    settings = get_settings()
    repository = _repository(user_id)
    file_name = (file.filename or "attachment")[:240]
    suffix = Path(file_name).suffix.lower()
    is_data_file = suffix in {".csv", ".xlsx", ".json", ".txt"}
    limit = (
        settings.assistant_data_file_max_bytes
        if is_data_file
        else settings.assistant_image_max_bytes
    )
    staging = repository.attachment_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{uuid4()}.upload"
    size = 0
    try:
        with temporary.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="数据文件不能超过 200MB。" if is_data_file else "图片不能超过 5MB。",
                    )
                target.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="上传文件为空。")
        width = height = 0
        if is_data_file:
            media_type = {
                ".csv": "text/csv",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".json": "application/json",
                ".txt": "text/plain",
            }[suffix]
            kind = "data_file"
        else:
            try:
                with Image.open(temporary) as image:
                    image.verify()
                with Image.open(temporary) as image:
                    media_type = Image.MIME.get(image.format or "", "")
                    width, height = image.size
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPException(status_code=415, detail="无法识别图片格式。") from exc
            if media_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(
                    status_code=415,
                    detail="仅支持 JPEG、PNG 和 WebP 图片，或 CSV、XLSX、JSON、TXT 数据文件。",
                )
            if max(width, height) > 4096:
                raise HTTPException(status_code=422, detail="图片最长边不能超过 4096 像素。")
            kind = "image"
        item = repository.save_attachment_file(
            conversation_id=conversation_id,
            file_name=file_name,
            media_type=media_type,
            source_path=temporary,
            attachment_kind=kind,
            width=width,
            height=height,
        )
        return _attachment_response(item)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)


@router.get("/attachments/{attachment_id}/content")
def get_attachment_content(
    attachment_id: UUID, user_id: str = Depends(current_user_id)
) -> FileResponse:
    repository = _repository(user_id)
    try:
        item = repository.get_attachment(attachment_id)
        return FileResponse(
            repository.attachment_path(attachment_id),
            media_type=str(item["media_type"]),
            filename=str(item["file_name"]),
            content_disposition_type="inline",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/import-batches/preview", response_model=AssistantImportBatchResponse, status_code=201
)
def preview_import_batch(
    request: AssistantImportBatchPreviewRequest, user_id: str = Depends(current_user_id)
) -> AssistantImportBatchResponse:
    repository = _repository(user_id)
    settings = get_settings()
    try:
        repository.get_conversation(request.conversation_id)
        if len(request.attachment_ids) > settings.assistant_data_file_max_count:
            raise ValueError("一次最多预览 20 个数据文件。")
        attachments = [repository.get_attachment(item) for item in request.attachment_ids]
        if any(str(item.get("attachment_kind")) != "data_file" for item in attachments):
            raise ValueError("导入批次只能包含数据文件。")
        if (
            sum(int(item["size_bytes"]) for item in attachments)
            > settings.assistant_data_batch_max_bytes
        ):
            raise ValueError("导入批次总量不能超过 1GB。")
        files = [_preview_data_attachment(repository, item) for item in attachments]
        batch = repository.create_import_batch(
            conversation_id=request.conversation_id,
            attachment_ids=request.attachment_ids,
            preview={
                "files": files,
                "valid_count": sum(1 for item in files if item["valid"]),
                "invalid_count": sum(1 for item in files if not item["valid"]),
            },
        )
        return AssistantImportBatchResponse.model_validate(batch)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/import-batches/{batch_id}/commit",
    response_model=AssistantImportBatchResponse,
    status_code=202,
)
def commit_import_batch(
    batch_id: UUID,
    request: AssistantImportBatchCommitRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantImportBatchResponse:
    repository = _repository(user_id)
    store = DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)
    try:
        batch = repository.get_import_batch(batch_id)
        if batch["status"] != "previewed":
            raise ValueError("Import batch is not ready to commit.")
        preview_files = list(batch["preview"].get("files") or [])
        if any(not item.get("valid") for item in preview_files) and not request.allow_partial:
            raise ValueError("部分文件预览失败，请移除失败文件或允许提交成功文件。")
        dataset_ids: list[UUID] = []
        attachment_map: dict[UUID, UUID] = {}
        for item in preview_files:
            if not item.get("valid"):
                continue
            attachment_id = UUID(str(item["attachment_id"]))
            attachment = repository.get_attachment(attachment_id)
            selected_sheet = request.sheet_selections.get(attachment_id) or item.get(
                "selected_sheet"
            )
            if item.get("requires_sheet_selection") and not selected_sheet:
                raise ValueError(f"请先为 {attachment['file_name']} 选择要导入的 Sheet。")
            dataset = None
            try:
                stream = record_batches_from_file_path(
                    repository.attachment_path(attachment_id),
                    source_type=str(item["source_type"]),
                    sheet_name=selected_sheet,
                )
                dataset = store.create_dataset(
                    name=str(attachment["file_name"]),
                    source_type=str(item["source_type"]),
                    source_metadata={
                        "kind": item["source_type"],
                        "name": attachment["file_name"],
                        "size_kb": round(int(attachment["size_bytes"]) / 1024, 1),
                        "parser": "assistant_tabular_import",
                        "streaming_import": True,
                        "sheet_name": stream.selected_sheet_name,
                        "assistant_import_batch_id": str(batch_id),
                    },
                )
                inserted = store.replace_raw_record_batches(
                    dataset_id=dataset.id,
                    batches=stream.batches,
                )
                if inserted <= 0:
                    raise ValueError(f"{attachment['file_name']} 中没有可导入记录。")
            except Exception:
                if dataset is not None:
                    with suppress(RuntimeError):
                        store.hard_delete_dataset(dataset.id)
                if request.allow_partial:
                    continue
                raise
            assert dataset is not None
            cleaning_job = store.create_cleaning_job(
                dataset_id=dataset.id,
                cleaning_strategy="auto",
                requirement="Kimi 数据包导入后的自动清洗",
            )
            start_cleaning_job(
                job_id=cleaning_job.id,
                user_id=user_id,
                dataset_store_path=get_settings().dataset_store_path,
            )
            dataset_ids.append(dataset.id)
            attachment_map[attachment_id] = dataset.id
        if not dataset_ids:
            raise ValueError("没有文件成功创建数据集。")
        group_id = None
        if len(dataset_ids) > 1:
            group = store.create_dataset_group(
                name=request.name or f"Kimi 数据包 {batch['created_at'][:10]}",
                dataset_ids=tuple(dataset_ids),
                description="由 Kimi 对话多文件导入创建。",
                metadata={"assistant_import_batch_id": str(batch_id)},
            )
            group_id = group.id
            suggestions = suggest_dataset_group_relationships(store, group_id=group.id)
            selection = select_automatic_dataset_relationships(group, suggestions.candidates)
            if selection.relationships:
                store.update_dataset_group_relationships(
                    group_id=group.id,
                    relationships=tuple(
                        item.model_dump(mode="json") for item in selection.relationships
                    ),
                )
        grant_type, grant_id = (
            ("dataset_group", group_id) if group_id else ("dataset", dataset_ids[0])
        )
        repository.save_permission_grant(
            asset_type=grant_type,
            asset_id=grant_id,
            capabilities=capabilities_for_asset(grant_type),
        )
        repository.update_conversation(
            batch["conversation_id"], scope_type=grant_type, scope_id=grant_id
        )
        completed = repository.complete_import_batch(
            batch_id,
            dataset_ids=tuple(dataset_ids),
            dataset_group_id=group_id,
            attachment_dataset_ids=attachment_map,
        )
        return AssistantImportBatchResponse.model_validate(completed)
    except (RuntimeError, ValueError) as exc:
        with suppress(RuntimeError):
            repository.fail_import_batch(batch_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=AssistantRunResponse,
    status_code=202,
)
def create_message(
    conversation_id: UUID,
    request: AssistantMessageCreateRequest,
    user_id: str = Depends(current_user_id),
) -> AssistantRunResponse:
    settings = get_settings()
    if not settings.assistant_enabled:
        raise HTTPException(status_code=503, detail="Kimi assistant is disabled.")
    if (
        not settings.kimi_api_key
        and settings.assistant_llm_provider == "kimi"
        and settings.environment != "test"
    ):
        raise HTTPException(status_code=503, detail="Kimi API Key 尚未配置。")
    try:
        enforce_rate_limit(
            f"assistant:{user_id}", limit=settings.assistant_rate_limit, window_seconds=60
        )
        repository = _repository(user_id)
        conversation = repository.get_conversation(conversation_id)
        if conversation.get("active_run_id"):
            raise HTTPException(status_code=409, detail="当前对话已有一条回复正在生成。")
        user_message = repository.create_message(
            conversation_id=conversation_id, role="user", content=request.content
        )
        repository.attach_to_message(
            message_id=user_message["message_id"], attachment_ids=request.attachment_ids
        )
        assistant_message = repository.create_message(
            conversation_id=conversation_id, role="assistant", content="", status="pending"
        )
        run = repository.create_run(
            conversation_id=conversation_id,
            user_message_id=user_message["message_id"],
            assistant_message_id=assistant_message["message_id"],
            execution_mode=request.execution_mode,
        )
        start_assistant_run(
            run_id=run.id, user_id=user_id, dataset_store_path=settings.dataset_store_path
        )
        return _run_response(run)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=AssistantRunResponse)
def get_run(run_id: UUID, user_id: str = Depends(current_user_id)) -> AssistantRunResponse:
    try:
        return _run_response(_repository(user_id).get_run(run_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: UUID,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    user_id: str = Depends(current_user_id),
) -> StreamingResponse:
    repository = _repository(user_id)
    try:
        repository.get_run(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        cursor = max(after_sequence, int(last_event_id or 0))
    except ValueError:
        cursor = after_sequence

    async def events():
        nonlocal cursor
        idle = 0
        terminal_grace = 0
        while True:
            rows = repository.list_events(run_id, after_sequence=cursor)
            for item in rows:
                cursor = int(item["sequence"])
                yield f"id: {cursor}\nevent: assistant\ndata: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"
                idle = 0
            run = repository.get_run(run_id)
            if run.status not in {
                "queued",
                "running",
                "pause_requested",
                "cancel_requested",
            } and not rows:
                maintenance = _memory_repository(user_id).get_maintenance_job_for_run(run_id)
                if maintenance and maintenance["status"] in {"queued", "running"}:
                    await asyncio.sleep(0.2)
                    continue
                if maintenance is None and terminal_grace < 2:
                    terminal_grace += 1
                    await asyncio.sleep(0.2)
                    continue
                yield f"event: end\ndata: {json.dumps({'status': run.status})}\n\n"
                return
            idle += 1
            if idle % 15 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.7)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/runs/{run_id}/confirm", response_model=AssistantRunResponse, status_code=202)
def confirm_run(
    run_id: UUID, request: AssistantRunConfirmRequest, user_id: str = Depends(current_user_id)
) -> AssistantRunResponse:
    repository = _repository(user_id)
    try:
        run = repository.get_run(run_id)
        if not request.accepted:
            confirmation_type = str(
                run.pending_confirmation.get("confirmation_type") or "analysis_plan"
            )
            message = (
                "已取消该操作。当前数据和资产均未发生变化。"
                if confirmation_type == "soft_delete"
                else "已取消该低置信度分析计划。请补充更明确的指标、维度或数据范围后再试。"
            )
            repository.update_message(
                run.assistant_message_id,
                content=message,
                status="completed",
                metadata={"confirmation_rejected": True, "confirmation_type": confirmation_type},
            )
            completed = repository.update_run(
                run_id,
                status="completed",
                current_stage="confirmation_rejected",
                pending_confirmation=run.pending_confirmation | {"accepted": False},
                completed=True,
            )
            repository.append_event(
                run_id, event_type="message.completed", status="completed", message=message
            )
            return _run_response(completed)
        queued = repository.confirm_run(run_id, accepted=True)
        start_assistant_run(
            run_id=run_id, user_id=user_id, dataset_store_path=get_settings().dataset_store_path
        )
        return _run_response(queued)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/cancel", response_model=AssistantRunResponse)
def cancel_run(run_id: UUID, user_id: str = Depends(current_user_id)) -> AssistantRunResponse:
    repository = _repository(user_id)
    try:
        run = repository.request_cancel(run_id)
        if run.analysis_job_id:
            DatasetStoreRepository(
                get_settings().dataset_store_path, user_id=user_id
            ).request_analysis_job_cancel(run.analysis_job_id)
        return _run_response(run)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/runs/{run_id}/pause", response_model=AssistantRunResponse, status_code=202)
def pause_run(run_id: UUID, user_id: str = Depends(current_user_id)) -> AssistantRunResponse:
    repository = _repository(user_id)
    try:
        return _run_response(repository.request_pause(run_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/resume", response_model=AssistantRunResponse, status_code=202)
def resume_run(run_id: UUID, user_id: str = Depends(current_user_id)) -> AssistantRunResponse:
    repository = _repository(user_id)
    try:
        resumed = repository.resume_run(run_id)
        start_assistant_run(
            run_id=run_id,
            user_id=user_id,
            dataset_store_path=get_settings().dataset_store_path,
        )
        return _run_response(resumed)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _run_response(run: StoredAssistantRun) -> AssistantRunResponse:
    return AssistantRunResponse(
        run_id=run.id,
        conversation_id=run.conversation_id,
        user_message_id=run.user_message_id,
        assistant_message_id=run.assistant_message_id,
        status=run.status,
        current_stage=run.current_stage,
        analysis_job_id=run.analysis_job_id,
        pending_confirmation=run.pending_confirmation,
        execution_mode=run.execution_mode,
        execution_plan=run.execution_plan,
        current_action_id=run.current_action_id,
        required_permission=run.required_permission,
        error=run.error,
        last_event_sequence=run.last_event_sequence,
        created_at=run.created_at,
        updated_at=run.updated_at,
        completed_at=run.completed_at,
    )


def _attachment_response(item: dict[str, Any]) -> AssistantAttachmentResponse:
    attachment_id = UUID(str(item["id"]))
    return AssistantAttachmentResponse(
        attachment_id=attachment_id,
        conversation_id=UUID(str(item["conversation_id"])),
        message_id=UUID(str(item["message_id"])) if item.get("message_id") else None,
        file_name=str(item["file_name"]),
        media_type=str(item["media_type"]),
        size_bytes=int(item["size_bytes"]),
        width=int(item["width"]),
        height=int(item["height"]),
        attachment_kind=str(item.get("attachment_kind") or "image"),
        import_status=str(item["import_status"]) if item.get("import_status") else None,
        dataset_id=UUID(str(item["dataset_id"])) if item.get("dataset_id") else None,
        import_batch_id=UUID(str(item["import_batch_id"])) if item.get("import_batch_id") else None,
        created_at=str(item["created_at"]),
        content_url=f"/assistant/attachments/{attachment_id}/content",
    )


def _message_response(
    repository: AssistantRepository, item: dict[str, Any]
) -> AssistantMessageResponse:
    deliverable_report_id = repository.deliverable_report_id_for_message(item["message_id"])
    citations = _citations_with_report_artifacts(
        repository,
        item.get("citations") or (),
        deliverable_report_id=deliverable_report_id,
    )
    return AssistantMessageResponse(
        message_id=item["message_id"],
        conversation_id=item["conversation_id"],
        role=item["role"],
        content=item["content"],
        status=item["status"],
        provider=item.get("provider"),
        model=item.get("model"),
        citations=citations,
        attachments=tuple(
            _attachment_response(value)
            for value in repository.list_message_attachments(item["message_id"])
        ),
        metadata=item.get("metadata") or {},
        created_at=item["created_at"],
    )


def _citations_with_report_artifacts(
    repository: AssistantRepository,
    citations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    deliverable_report_id: UUID | None = None,
) -> tuple[dict[str, Any], ...]:
    enriched = [dict(item) | {"artifact_role": item.get("artifact_role") or "evidence"} for item in citations]
    if deliverable_report_id is not None:
        for item in enriched:
            if item.get("source_type") == "report" and str(item.get("source_id")) == str(
                deliverable_report_id
            ):
                item["artifact_role"] = "deliverable"
    known = {(str(item.get("source_type")), str(item.get("source_id"))) for item in enriched}
    store = DatasetStoreRepository(get_settings().dataset_store_path, user_id=repository.user_id)
    for citation in tuple(enriched):
        if citation.get("source_type") != "analysis_job" or not citation.get("source_id"):
            continue
        try:
            job = store.get_analysis_job(UUID(str(citation["source_id"])))
            job_result = job.result if isinstance(job.result, dict) else {}
            citation["reliability"] = canonical_reliability(
                citation.get("reliability"),
                job_result.get("statistical_verification"),
            )
            if job.report_id is None:
                continue
            report = store.get_report(job.report_id)
        except (RuntimeError, ValueError):
            continue
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        structured = (
            metadata.get("structured_report")
            if isinstance(metadata.get("structured_report"), dict)
            else {}
        )
        excerpt = safe_excerpt(
            structured.get("executive_summary") or report.get("markdown") or "完整分析报告已生成。"
        )
        report_citation = next(
            (
                item
                for item in enriched
                if item.get("source_type") == "report"
                and str(item.get("source_id")) == str(job.report_id)
            ),
            None,
        )
        lineage_reliability = canonical_reliability(
            citation.get("reliability"),
            report_citation.get("reliability") if report_citation else None,
            metadata.get("statistical_verification"),
        )
        citation["reliability"] = lineage_reliability
        if report_citation is None:
            report_citation = {
                "source_type": "report",
                "source_id": str(job.report_id),
                "label": str(report["title"]),
                "excerpt": excerpt,
                "dataset_id": str(report["dataset_id"]),
                "artifact_role": "evidence",
            }
            enriched.append(report_citation)
        else:
            report_citation["excerpt"] = safe_excerpt(report_citation.get("excerpt"))
        report_citation["reliability"] = lineage_reliability
        if (
            citation.get("artifact_role") == "deliverable"
            or str(job.report_id) == str(deliverable_report_id)
        ):
            report_citation["artifact_role"] = "deliverable"
        known.add(("report", str(job.report_id)))
    for item in enriched:
        item["excerpt"] = safe_excerpt(item.get("excerpt"))
    deliverable_indexes = [
        index
        for index, item in enumerate(enriched)
        if item.get("source_type") == "report" and item.get("artifact_role") == "deliverable"
    ]
    for index in deliverable_indexes[:-1]:
        enriched[index]["artifact_role"] = "evidence"
    public_fields = {
        "source_type",
        "source_id",
        "label",
        "excerpt",
        "dataset_id",
        "artifact_role",
        "reliability",
    }
    # Older or interrupted runs may contain server-only enrichment fields.
    # Sanitize on read as well as write so one malformed citation cannot make
    # the entire conversation endpoint return 500.
    return tuple(
        {key: value for key, value in item.items() if key in public_fields}
        for item in enriched
    )


def _preview_data_attachment(
    repository: AssistantRepository, attachment: dict[str, Any]
) -> dict[str, Any]:
    attachment_id = UUID(str(attachment["id"]))
    suffix = Path(str(attachment["file_name"])).suffix.lower()
    source_type = suffix.removeprefix(".")
    base = {
        "attachment_id": str(attachment_id),
        "file_name": str(attachment["file_name"]),
        "source_type": source_type,
        "valid": False,
        "error": None,
        "row_count": 0,
        "column_count": 0,
        "columns": [],
        "preview_records": [],
        "selected_sheet": None,
        "sheets": [],
    }
    try:
        file_path = repository.attachment_path(attachment_id)
        if source_type == "xlsx":
            sheets = xlsx_sheet_previews_from_path(file_path)
            if not sheets.get("ok"):
                return base | {"error": str(sheets.get("error") or "XLSX 预览失败。")}
            sheet_items = list(sheets.get("sheets") or [])
            selected = next((item for item in sheet_items if item.get("selected")), sheet_items[0])
            top_score = int(selected.get("score") or 0)
            second_score = int(sheet_items[1].get("score") or 0) if len(sheet_items) > 1 else 0
            requires_selection = bool(
                second_score and top_score and second_score / top_score >= 0.9
            )
            preview = list(selected.get("preview_records") or [])
            return base | {
                "valid": True,
                "row_count": int(selected.get("row_count") or 0),
                "column_count": int(selected.get("column_count") or 0),
                "columns": list(preview[0].keys()) if preview else [],
                "preview_records": preview,
                "selected_sheet": None if requires_selection else str(selected["sheet_name"]),
                "requires_sheet_selection": requires_selection,
                "sheets": sheet_items,
            }
        parsed = preview_file_from_path(file_path, source_type=source_type)
        if not parsed.get("ok"):
            return base | {"error": str(parsed.get("error") or "文件中没有可导入记录。")}
        return base | {
            "valid": True,
            "row_count": int(parsed.get("row_count") or 0),
            "column_count": int(parsed.get("column_count") or 0),
            "columns": list(parsed.get("columns") or []),
            "preview_records": list(parsed.get("preview_records") or []),
        }
    except Exception as exc:
        return base | {"error": str(exc)}


def _undo_action(store: DatasetStoreRepository, action: dict[str, Any]) -> None:
    tool_name = str(action["tool_name"])
    before = dict(action.get("before_state") or {})
    asset_id = action.get("asset_id")
    if tool_name in {"activate_cleaning_version", "rollback_cleaning_version"}:
        run_id = before.get("run_id")
        if not asset_id or not run_id:
            raise ValueError("缺少可恢复的清洗版本。")
        store.activate_cleaning_run(dataset_id=asset_id, run_id=UUID(str(run_id)))
    elif tool_name == "update_column_metadata":
        column = before
        if not asset_id or not column.get("column_name"):
            raise ValueError("缺少可恢复的字段元数据。")
        store.update_column_metadata(
            dataset_id=asset_id,
            column_name=str(column["column_name"]),
            override_type=column.get("override_type"),
            role=column.get("role"),
            description=column.get("description"),
        )
    elif tool_name == "save_relationship_plan":
        if not asset_id:
            raise ValueError("缺少数据包标识。")
        store.update_dataset_group_relationships(
            group_id=asset_id,
            relationships=tuple(dict(item) for item in before.get("relationships") or []),
        )
    elif tool_name == "rename_report":
        if not asset_id:
            raise ValueError("缺少报告标识。")
        store.update_report(report_id=asset_id, title=str(before.get("title") or ""))
    elif tool_name == "revise_report":
        created_report_id = dict(action.get("after_state") or {}).get("created_report_id")
        if not created_report_id:
            raise ValueError("缺少新报告标识。")
        store.delete_report(UUID(str(created_report_id)))
    elif tool_name == "soft_delete_asset":
        if not action.get("asset_type") or not asset_id:
            raise ValueError("缺少回收站资产标识。")
        store.restore_asset(asset_type=str(action["asset_type"]), asset_id=asset_id)
    elif tool_name == "update_semantic_draft":
        if not asset_id or not before.get("definition"):
            raise ValueError("缺少可恢复的语义模型定义。")
        current = store.get_semantic_model(asset_id)
        from app.semantic.service import SemanticLayerService

        SemanticLayerService(store).update_draft(
            asset_id,
            revision=int(current["revision"]),
            name=str(before.get("name") or current["name"]),
            definition=dict(before["definition"]),
        )
    else:
        raise ValueError("当前操作暂不支持自动撤销。")
