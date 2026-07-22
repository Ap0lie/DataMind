from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.analysis.cleaning_jobs import start_cleaning_job
from app.analysis.data_cleaning import DataCleaningService
from app.analysis.dataset_groups import (
    select_automatic_dataset_relationships,
    suggest_dataset_group_relationships,
)
from app.analysis.services import DatasetProfiler, apply_column_metadata_to_profile
from app.api.v1.deps import current_user_id
from app.core.settings import get_settings
from app.schemas.analysis import DatasetProfileResponse
from app.schemas.dataset_store import (
    AppendRawRecordsRequest,
    AppendRawRecordsResponse,
    CleaningDiffSummary,
    CleaningJobResponse,
    CleaningRulePreviewRequest,
    CleaningRulePreviewResponse,
    CleaningRunDetail,
    CreateCleaningJobRequest,
    CreateDatasetGroupRequest,
    CreateDatasetRequest,
    CreateDatasetResponse,
    DatasetCleaningRunListResponse,
    DatasetCleaningRunResponse,
    DatasetColumnMetadata,
    DatasetColumnMetadataListResponse,
    DatasetGroupListResponse,
    DatasetGroupResponse,
    DatasetGroupTable,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetRelationshipAutoConfigureResponse,
    DatasetRelationshipPlan,
    DatasetRelationshipSuggestionResponse,
    DatasetReportListResponse,
    DatasetReportResponse,
    DeleteDatasetGroupResponse,
    DeleteDatasetResponse,
    DeleteReportResponse,
    ExcelSheetPreview,
    ExcelSheetPreviewResponse,
    FileDatasetImportResponse,
    ReportUpdateRequest,
    ReportVersionListResponse,
    ReportVersionSummary,
    RunDatasetCleaningRequest,
    SaveArtifactResponse,
    SaveChartRequest,
    SaveDatasetArtifactRequest,
    SaveDatasetColumnsRequest,
    SaveReportRequest,
    UpdateDatasetColumnRequest,
    UpdateDatasetGroupRelationshipsRequest,
)
from app.schemas.semantic import (
    SemanticModelCopyRequest,
    SemanticModelDraftRequest,
    SemanticModelListResponse,
    SemanticModelResponse,
    SemanticModelUpdateRequest,
    SemanticValidationResponse,
)
from app.semantic.service import SemanticLayerService
from app.services.cleaning_rules import apply_cleaning_rules
from app.services.tabular_import import records_from_file_bytes, xlsx_sheet_previews_from_bytes
from app.storage.dataset_store import DatasetStoreRepository, StoredDataset, StoredDatasetGroup

router = APIRouter()


@router.post("/semantic-models/drafts", response_model=SemanticModelResponse)
def create_semantic_model_draft(
    request: SemanticModelDraftRequest, user_id: str = Depends(current_user_id)
) -> SemanticModelResponse:
    try:
        model = SemanticLayerService(_repository(user_id)).create_draft(
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            name=request.name,
            source_model_id=request.source_model_id,
        )
        return _semantic_model_response(model)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/semantic-models", response_model=SemanticModelListResponse)
def list_semantic_models(
    scope_type: str, scope_id: UUID, user_id: str = Depends(current_user_id)
) -> SemanticModelListResponse:
    if scope_type not in {"dataset", "dataset_group"}:
        raise HTTPException(status_code=400, detail="Invalid semantic scope type.")
    models = _repository(user_id).list_semantic_models(scope_type=scope_type, scope_id=scope_id)
    return SemanticModelListResponse(
        models=tuple(_semantic_model_response(item) for item in models)
    )


@router.get("/semantic-models/{model_id}", response_model=SemanticModelResponse)
def get_semantic_model(
    model_id: UUID, user_id: str = Depends(current_user_id)
) -> SemanticModelResponse:
    try:
        return _semantic_model_response(_repository(user_id).get_semantic_model(model_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/semantic-models/{model_id}", response_model=SemanticModelResponse)
def update_semantic_model(
    model_id: UUID, request: SemanticModelUpdateRequest, user_id: str = Depends(current_user_id)
) -> SemanticModelResponse:
    try:
        model = SemanticLayerService(_repository(user_id)).update_draft(
            model_id,
            revision=request.revision,
            name=request.name,
            definition=request.definition,
        )
        return _semantic_model_response(model)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/semantic-models/{model_id}/validate", response_model=SemanticValidationResponse)
def validate_semantic_model(
    model_id: UUID, user_id: str = Depends(current_user_id)
) -> SemanticValidationResponse:
    repository = _repository(user_id)
    try:
        return SemanticValidationResponse.model_validate(
            SemanticLayerService(repository).validate(repository.get_semantic_model(model_id))
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/semantic-models/{model_id}/publish", response_model=SemanticModelResponse)
def publish_semantic_model(
    model_id: UUID, user_id: str = Depends(current_user_id)
) -> SemanticModelResponse:
    try:
        return _semantic_model_response(
            SemanticLayerService(_repository(user_id)).publish(model_id)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/semantic-models/{model_id}/copy", response_model=SemanticModelResponse)
def copy_semantic_model(
    model_id: UUID, request: SemanticModelCopyRequest, user_id: str = Depends(current_user_id)
) -> SemanticModelResponse:
    try:
        model = SemanticLayerService(_repository(user_id)).create_draft(
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            name=request.name,
            source_model_id=model_id,
        )
        return _semantic_model_response(model)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(user_id: str = Depends(current_user_id)) -> DatasetListResponse:
    try:
        datasets = _repository(user_id).list_datasets()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DatasetListResponse(datasets=tuple(_dataset_response(dataset) for dataset in datasets))


@router.post("/datasets", response_model=CreateDatasetResponse)
def create_dataset(
    request: CreateDatasetRequest,
    user_id: str = Depends(current_user_id),
) -> CreateDatasetResponse:
    try:
        dataset = _repository(user_id).create_dataset(
            name=request.name,
            source_type=request.source_type,
            source_metadata=request.source_metadata,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _dataset_response(dataset)


@router.post("/dataset-groups", response_model=DatasetGroupResponse)
def create_dataset_group(
    request: CreateDatasetGroupRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetGroupResponse:
    try:
        repository = _repository(user_id)
        group = repository.create_dataset_group(
            name=request.name,
            dataset_ids=request.dataset_ids,
            description=request.description,
            metadata=request.metadata,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _dataset_group_response(repository, group)


@router.get("/dataset-groups", response_model=DatasetGroupListResponse)
def list_dataset_groups(user_id: str = Depends(current_user_id)) -> DatasetGroupListResponse:
    try:
        repository = _repository(user_id)
        groups = repository.list_dataset_groups()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DatasetGroupListResponse(
        groups=tuple(_dataset_group_response(repository, group) for group in groups)
    )


@router.get("/dataset-groups/{group_id}", response_model=DatasetGroupResponse)
def get_dataset_group(
    group_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetGroupResponse:
    try:
        repository = _repository(user_id)
        group = repository.get_dataset_group(group_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _dataset_group_response(repository, group)


@router.post(
    "/dataset-groups/{group_id}/relationship-suggestions",
    response_model=DatasetRelationshipSuggestionResponse,
)
def suggest_dataset_group_relationships_api(
    group_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetRelationshipSuggestionResponse:
    try:
        repository = _repository(user_id)
        suggestions = suggest_dataset_group_relationships(repository, group_id=group_id)
        group = repository.get_dataset_group(group_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetRelationshipSuggestionResponse(
        group=_dataset_group_response(repository, group),
        candidates=suggestions.candidates,
        llm_used=suggestions.llm_used,
        compact_context=suggestions.compact_context,
        validation_issues=suggestions.validation_issues,
    )


@router.patch("/dataset-groups/{group_id}/relationships", response_model=DatasetGroupResponse)
def update_dataset_group_relationships(
    group_id: UUID,
    request: UpdateDatasetGroupRelationshipsRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetGroupResponse:
    try:
        repository = _repository(user_id)
        group = repository.update_dataset_group_relationships(
            group_id=group_id,
            relationships=tuple(item.model_dump(mode="json") for item in request.relationships),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _dataset_group_response(repository, group)


@router.post(
    "/dataset-groups/{group_id}/relationships/auto-configure",
    response_model=DatasetRelationshipAutoConfigureResponse,
)
def auto_configure_dataset_group_relationships(
    group_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetRelationshipAutoConfigureResponse:
    try:
        repository = _repository(user_id)
        suggestions = suggest_dataset_group_relationships(repository, group_id=group_id)
        stored_group = repository.get_dataset_group(group_id)
        selection = select_automatic_dataset_relationships(stored_group, suggestions.candidates)
        updated_group = repository.update_dataset_group_relationships(
            group_id=group_id,
            relationships=tuple(item.model_dump(mode="json") for item in selection.relationships),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    issues = list(suggestions.validation_issues)
    if selection.unresolved_dataset_ids:
        issues.append(
            f"{len(selection.unresolved_dataset_ids)} table(s) had no reliable relationship and were not auto-connected."
        )
    if not selection.relationships:
        issues.append(
            "No relationship passed automatic validation; the dataset group was left unchanged."
        )
    return DatasetRelationshipAutoConfigureResponse(
        group=_dataset_group_response(repository, updated_group),
        candidates=suggestions.candidates,
        llm_used=suggestions.llm_used,
        compact_context=suggestions.compact_context,
        validation_issues=tuple(issues),
        saved_relationships=selection.relationships,
        primary_dataset_id=selection.primary_dataset_id,
        unresolved_dataset_ids=selection.unresolved_dataset_ids,
    )


@router.delete("/dataset-groups/{group_id}", response_model=DeleteDatasetGroupResponse)
def delete_dataset_group(
    group_id: UUID,
    delete_datasets: bool = True,
    user_id: str = Depends(current_user_id),
) -> DeleteDatasetGroupResponse:
    try:
        deleted_dataset_ids = _repository(user_id).delete_dataset_group(
            group_id,
            delete_datasets=delete_datasets,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteDatasetGroupResponse(
        group_id=group_id,
        deleted=True,
        deleted_dataset_ids=deleted_dataset_ids,
    )


@router.get("/datasets/{dataset_id}", response_model=CreateDatasetResponse)
def get_dataset(
    dataset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> CreateDatasetResponse:
    try:
        dataset = _repository(user_id).get_dataset(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _dataset_response(dataset)


@router.delete("/datasets/{dataset_id}", response_model=DeleteDatasetResponse)
def delete_dataset(
    dataset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DeleteDatasetResponse:
    try:
        _repository(user_id).delete_dataset(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteDatasetResponse(dataset_id=dataset_id, deleted=True)


@router.post("/datasets/{dataset_id}/raw-records", response_model=AppendRawRecordsResponse)
def append_raw_records(
    dataset_id: UUID,
    request: AppendRawRecordsRequest,
    user_id: str = Depends(current_user_id),
) -> AppendRawRecordsResponse:
    try:
        inserted = _repository(user_id).append_raw_records(
            dataset_id=dataset_id, records=request.records
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AppendRawRecordsResponse(dataset_id=dataset_id, inserted=inserted)


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset(
    dataset_id: UUID,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    source: Annotated[str, Query(pattern="^(raw|cleaned|analysis)$")] = "raw",
    user_id: str = Depends(current_user_id),
) -> DatasetPreviewResponse:
    try:
        repository = _repository(user_id)
        if source == "cleaned":
            records = repository.preview_cleaned_records(dataset_id, limit=limit)
        elif source == "analysis":
            records = repository.preview_analysis_records(dataset_id, limit=limit)
        else:
            records = repository.preview_raw_records(dataset_id, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetPreviewResponse(
        dataset_id=dataset_id, record_source=source, records=tuple(records)
    )


@router.post("/files/import", response_model=FileDatasetImportResponse)
async def import_dataset_file(
    file: UploadFile = File(...),
    dataset_name: str | None = Form(default=None),
    sheet_name: str | None = Form(default=None),
    user_id: str = Depends(current_user_id),
) -> FileDatasetImportResponse:
    file_name = file.filename or "uploaded_dataset"
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        source_type = "csv"
    elif suffix == ".xlsx":
        source_type = "xlsx"
    elif suffix == ".json":
        source_type = "json"
    elif suffix == ".txt":
        source_type = "txt"
    else:
        raise HTTPException(
            status_code=400, detail="Only CSV, XLSX, JSON, and TXT files can be imported."
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    parsed = records_from_file_bytes(
        file_bytes=file_bytes,
        source_type=source_type,
        sheet_name=sheet_name if source_type == "xlsx" else None,
    )
    if not parsed.get("ok"):
        raise HTTPException(
            status_code=400, detail=str(parsed.get("error") or "File parsing failed.")
        )
    records = parsed.get("data")
    if not isinstance(records, list) or not records:
        raise HTTPException(
            status_code=400, detail="No tabular records were parsed from the uploaded file."
        )
    try:
        repository = _repository(user_id)
        dataset = repository.create_dataset(
            name=dataset_name or file_name,
            source_type=source_type,
            source_metadata={
                "kind": source_type,
                "name": file_name,
                "size_kb": round(len(file_bytes) / 1024, 1),
                "parser": "backend_tabular_import",
                "sheet_name": sheet_name,
            },
        )
        inserted = repository.append_raw_records(dataset_id=dataset.id, records=records)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileDatasetImportResponse(
        dataset=_dataset_response(dataset),
        inserted=inserted,
        preview_records=tuple(records[:50]),
    )


@router.post("/files/xlsx-sheets", response_model=ExcelSheetPreviewResponse)
async def preview_xlsx_sheets(
    file: UploadFile = File(...),
) -> ExcelSheetPreviewResponse:
    file_name = file.filename or "uploaded.xlsx"
    if Path(file_name).suffix.lower() != ".xlsx":
        raise HTTPException(status_code=400, detail="Only XLSX files support sheet preview.")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    parsed = xlsx_sheet_previews_from_bytes(file_bytes)
    if not parsed.get("ok"):
        raise HTTPException(
            status_code=400, detail=str(parsed.get("error") or "XLSX sheet parsing failed.")
        )
    sheets = parsed.get("sheets")
    if not isinstance(sheets, list):
        raise HTTPException(status_code=400, detail="No XLSX sheets were parsed.")
    return ExcelSheetPreviewResponse(
        sheets=tuple(
            ExcelSheetPreview(
                sheet_name=str(sheet.get("sheet_name") or ""),
                row_count=int(sheet.get("row_count") or 0),
                column_count=int(sheet.get("column_count") or 0),
                score=int(sheet.get("score") or 0),
                selected=bool(sheet.get("selected")),
                preview_records=tuple(
                    record
                    for record in sheet.get("preview_records", [])
                    if isinstance(record, dict)
                ),
            )
            for sheet in sheets
            if isinstance(sheet, dict)
        )
    )


@router.get("/reports", response_model=DatasetReportListResponse)
def list_reports(
    include_content: bool = True,
    dataset_id: UUID | None = None,
    query: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    user_id: str = Depends(current_user_id),
) -> DatasetReportListResponse:
    try:
        reports = _repository(user_id).list_reports(
            dataset_id=dataset_id,
            query=query,
            limit=limit,
            include_content=include_content,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DatasetReportListResponse(
        reports=tuple(
            _report_response(report, include_content=include_content) for report in reports
        )
    )


@router.get("/datasets/{dataset_id}/reports", response_model=DatasetReportListResponse)
def list_dataset_reports(
    dataset_id: UUID,
    include_content: bool = True,
    user_id: str = Depends(current_user_id),
) -> DatasetReportListResponse:
    try:
        reports = _repository(user_id).list_reports(
            dataset_id=dataset_id,
            include_content=include_content,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetReportListResponse(
        reports=tuple(
            _report_response(report, include_content=include_content) for report in reports
        )
    )


@router.get("/reports/{report_id}", response_model=DatasetReportResponse)
def get_report(
    report_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetReportResponse:
    try:
        report = _repository(user_id).get_report(report_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _report_response(report)


@router.patch("/reports/{report_id}", response_model=DatasetReportResponse)
def update_report(
    report_id: UUID,
    request: ReportUpdateRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetReportResponse:
    try:
        report = _repository(user_id).update_report(report_id=report_id, title=request.title)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _report_response(report)


@router.delete("/reports/{report_id}", response_model=DeleteReportResponse)
def delete_report(
    report_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DeleteReportResponse:
    try:
        _repository(user_id).delete_report(report_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteReportResponse(report_id=report_id, deleted=True)


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfileResponse)
def profile_dataset(
    dataset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetProfileResponse:
    try:
        repository = _repository(user_id)
        records = repository.read_analysis_records(dataset_id)
        metadata = repository.list_column_metadata(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    profile = DatasetProfiler().profile(dataset_id=dataset_id, records=records)
    return apply_column_metadata_to_profile(profile, metadata)


@router.get("/datasets/{dataset_id}/columns", response_model=DatasetColumnMetadataListResponse)
def list_dataset_columns(
    dataset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetColumnMetadataListResponse:
    try:
        columns = _repository(user_id).list_column_metadata(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetColumnMetadataListResponse(
        columns=tuple(_column_response(item) for item in columns)
    )


@router.post("/datasets/{dataset_id}/columns", response_model=DatasetColumnMetadataListResponse)
def save_dataset_columns(
    dataset_id: UUID,
    request: SaveDatasetColumnsRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetColumnMetadataListResponse:
    try:
        columns = _repository(user_id).save_column_metadata(
            dataset_id=dataset_id,
            columns=[column.model_dump(mode="json") for column in request.columns],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetColumnMetadataListResponse(
        columns=tuple(_column_response(item) for item in columns)
    )


@router.patch("/datasets/{dataset_id}/columns/{column_name}", response_model=DatasetColumnMetadata)
def update_dataset_column(
    dataset_id: UUID,
    column_name: str,
    request: UpdateDatasetColumnRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetColumnMetadata:
    try:
        column = _repository(user_id).update_column_metadata(
            dataset_id=dataset_id,
            column_name=column_name,
            inferred_type=request.inferred_type,
            override_type=request.override_type,
            role=request.role,
            description=request.description,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _column_response(column)


@router.post("/datasets/{dataset_id}/cleaning-jobs", response_model=CleaningJobResponse)
def create_cleaning_job(
    dataset_id: UUID,
    request: CreateCleaningJobRequest,
    user_id: str = Depends(current_user_id),
) -> CleaningJobResponse:
    try:
        repository = _repository(user_id)
        job = repository.create_cleaning_job(
            dataset_id=dataset_id,
            requirement=request.requirement,
            cleaning_strategy=request.cleaning_strategy,
            prompt_overrides=request.prompt_overrides.as_dict(),
        )
        start_cleaning_job(
            job_id=job.id,
            user_id=user_id,
            dataset_store_path=repository.root_path,
        )
        return _cleaning_job_response(repository.get_cleaning_job(job.id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/cleaning-jobs/{job_id}", response_model=CleaningJobResponse)
def get_cleaning_job(
    dataset_id: UUID, job_id: UUID, user_id: str = Depends(current_user_id)
) -> CleaningJobResponse:
    try:
        job = _repository(user_id).get_cleaning_job(job_id)
        if job.dataset_id != dataset_id:
            raise RuntimeError(f"Cleaning job was not found: {job_id}")
        return _cleaning_job_response(job)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/cleaning-jobs/{job_id}/events")
async def stream_cleaning_job_events(
    dataset_id: UUID,
    job_id: UUID,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    user_id: str = Depends(current_user_id),
) -> StreamingResponse:
    repository = _repository(user_id)
    try:
        job = repository.get_cleaning_job(job_id)
        if job.dataset_id != dataset_id:
            raise RuntimeError(f"Cleaning job was not found: {job_id}")
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        cursor = max(after_sequence, int(last_event_id or 0))
    except ValueError:
        cursor = after_sequence

    async def event_stream():
        nonlocal cursor
        idle_ticks = 0
        while True:
            events = repository.list_cleaning_job_events(job_id, after_sequence=cursor)
            for event in events:
                cursor = int(event.get("sequence") or cursor)
                yield f"id: {cursor}\nevent: cleaning\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                idle_ticks = 0
            current = repository.get_cleaning_job(job_id)
            if current.status not in {"queued", "running", "cancel_requested"} and not events:
                yield f"event: end\ndata: {json.dumps({'status': current.status})}\n\n"
                return
            idle_ticks += 1
            if idle_ticks % 15 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post(
    "/datasets/{dataset_id}/cleaning-jobs/{job_id}/cancel", response_model=CleaningJobResponse
)
def cancel_cleaning_job(
    dataset_id: UUID, job_id: UUID, user_id: str = Depends(current_user_id)
) -> CleaningJobResponse:
    repository = _repository(user_id)
    try:
        job = repository.get_cleaning_job(job_id)
        if job.dataset_id != dataset_id:
            raise RuntimeError(f"Cleaning job was not found: {job_id}")
        return _cleaning_job_response(repository.request_cleaning_job_cancel(job_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/datasets/{dataset_id}/cleaning-jobs/{job_id}/retry", response_model=CleaningJobResponse
)
def retry_cleaning_job(
    dataset_id: UUID, job_id: UUID, user_id: str = Depends(current_user_id)
) -> CleaningJobResponse:
    repository = _repository(user_id)
    try:
        previous = repository.get_cleaning_job(job_id)
        if previous.dataset_id != dataset_id:
            raise RuntimeError(f"Cleaning job was not found: {job_id}")
        if previous.status in {"queued", "running", "cancel_requested"}:
            raise ValueError("A running cleaning job cannot be retried.")
        job = repository.create_cleaning_job(
            dataset_id=dataset_id,
            requirement=previous.requirement,
            cleaning_strategy=previous.cleaning_strategy,
            prompt_overrides=previous.prompt_overrides,
            retry_of=previous.id,
        )
        start_cleaning_job(job_id=job.id, user_id=user_id, dataset_store_path=repository.root_path)
        return _cleaning_job_response(repository.get_cleaning_job(job.id))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/datasets/{dataset_id}/cleaning-jobs/{job_id}/result",
    response_model=DatasetCleaningRunResponse,
)
def get_cleaning_job_result(
    dataset_id: UUID, job_id: UUID, user_id: str = Depends(current_user_id)
) -> DatasetCleaningRunResponse:
    repository = _repository(user_id)
    try:
        job = repository.get_cleaning_job(job_id)
        if job.dataset_id != dataset_id:
            raise RuntimeError(f"Cleaning job was not found: {job_id}")
        if job.status != "completed" or job.cleaning_run_id is None:
            raise ValueError("Cleaning result is not ready.")
        run = repository.get_cleaning_run(dataset_id, job.cleaning_run_id)
        result = job.result or {}
        cleaned = run.get("cleaned_dataset") if isinstance(run.get("cleaned_dataset"), dict) else {}
        return DatasetCleaningRunResponse(
            dataset_id=dataset_id,
            run_id=job.cleaning_run_id,
            version=int(run.get("version") or 1),
            provider=str(run.get("provider") or "rules"),
            model=str(run.get("model") or "local-basic-cleaner"),
            source=str(result.get("selected_strategy") or cleaned.get("source") or "rules"),
            raw_row_count=int((run.get("raw_summary") or {}).get("row_count") or 0),
            cleaned_row_count=int(result.get("row_count") or cleaned.get("rows") or 0),
            cleaned_column_count=int(result.get("column_count") or cleaned.get("columns") or 0),
            result_markdown=str(run.get("result_markdown") or ""),
            preview_records=tuple(
                item for item in result.get("preview_records") or () if isinstance(item, dict)
            ),
            warnings=tuple(str(item.get("error") or item) for item in result.get("failures") or ()),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/cleaning-runs", response_model=DatasetCleaningRunListResponse)
def list_cleaning_runs(
    dataset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> DatasetCleaningRunListResponse:
    try:
        runs = _repository(user_id).list_cleaning_runs(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetCleaningRunListResponse(runs=tuple(_cleaning_run_response(run) for run in runs))


@router.get("/datasets/{dataset_id}/cleaning-runs/{run_id}", response_model=CleaningRunDetail)
def get_cleaning_run(
    dataset_id: UUID,
    run_id: UUID,
    user_id: str = Depends(current_user_id),
) -> CleaningRunDetail:
    try:
        run = _repository(user_id).get_cleaning_run(dataset_id, run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _cleaning_run_response(run)


@router.post(
    "/datasets/{dataset_id}/cleaning-runs/{run_id}/activate", response_model=CleaningRunDetail
)
def activate_cleaning_run(
    dataset_id: UUID,
    run_id: UUID,
    user_id: str = Depends(current_user_id),
) -> CleaningRunDetail:
    try:
        run = _repository(user_id).activate_cleaning_run(dataset_id=dataset_id, run_id=run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _cleaning_run_response(run)


@router.post(
    "/datasets/{dataset_id}/cleaning-rules/preview", response_model=CleaningRulePreviewResponse
)
def preview_cleaning_rules(
    dataset_id: UUID,
    request: CleaningRulePreviewRequest,
    user_id: str = Depends(current_user_id),
) -> CleaningRulePreviewResponse:
    try:
        repository = _repository(user_id)
        base_records = repository.read_analysis_records(dataset_id)
        current_records, issues, applied_rules = apply_cleaning_rules(
            base_records,
            [rule.model_dump(mode="json") for rule in request.rules],
        )
        raw_records = repository.read_raw_records(dataset_id)
        diff_summary = _cleaning_diff_summary(
            raw_records=raw_records,
            previous_records=base_records,
            current_records=current_records,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CleaningRulePreviewResponse(
        dataset_id=dataset_id,
        preview_records=tuple(current_records[:50]),
        diff_summary=_diff_response(diff_summary),
        validation_issues=tuple(issues),
        applied_rules=tuple(
            request.rules[index]
            for index, rule in enumerate(request.rules)
            if index < len(request.rules)
            and any(rule.model_dump(mode="json") == applied for applied in applied_rules)
        ),
    )


@router.post(
    "/datasets/{dataset_id}/cleaning-rules/apply", response_model=DatasetCleaningRunResponse
)
def apply_cleaning_rule_set(
    dataset_id: UUID,
    request: CleaningRulePreviewRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetCleaningRunResponse:
    try:
        repository = _repository(user_id)
        raw_records = repository.read_raw_records(dataset_id)
        previous_records = repository.read_analysis_records(dataset_id)
        cleaned_records, issues, applied_rules = apply_cleaning_rules(
            previous_records,
            [rule.model_dump(mode="json") for rule in request.rules],
        )
        raw_summary = _record_summary(raw_records)
        previous_summary = _record_summary(previous_records)
        current_summary = _record_summary(cleaned_records)
        diff_summary = _cleaning_diff_summary(
            raw_records=raw_records,
            previous_records=previous_records,
            current_records=cleaned_records,
        )
        cleaned_count = repository.save_cleaned_records(
            dataset_id=dataset_id,
            records=cleaned_records,
            metadata={
                "provider": "rules",
                "model": "local_cleaning_rules",
                "source": "manual_rules",
                "rules": applied_rules,
                "warnings": issues,
            },
        )
        run_id = repository.save_cleaning_result(
            dataset_id=dataset_id,
            provider="rules",
            model="local_cleaning_rules",
            prompt="手动清洗规则",
            result_markdown=_cleaning_rules_markdown(applied_rules, issues),
            cleaned_dataset={
                "status": "completed",
                "source": "manual_rules",
                "rows": cleaned_count,
                "columns": len(cleaned_records[0]) if cleaned_records else 0,
                "warnings": issues,
                "records": cleaned_records,
            },
            raw_summary=raw_summary,
            previous_summary=previous_summary,
            current_summary=current_summary,
            diff_summary=diff_summary,
        )
        run = repository.get_cleaning_run(dataset_id, run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetCleaningRunResponse(
        dataset_id=dataset_id,
        run_id=run_id,
        version=int(run.get("version") or 1),
        provider="rules",
        model="local_cleaning_rules",
        source="manual_rules",
        raw_row_count=len(raw_records),
        cleaned_row_count=cleaned_count,
        cleaned_column_count=len(cleaned_records[0]) if cleaned_records else 0,
        result_markdown=_cleaning_rules_markdown(applied_rules, issues),
        preview_records=tuple(cleaned_records[:50]),
        warnings=tuple(issues),
    )


@router.post("/datasets/{dataset_id}/artifacts", response_model=SaveArtifactResponse)
def save_dataset_artifact(
    dataset_id: UUID,
    request: SaveDatasetArtifactRequest,
    user_id: str = Depends(current_user_id),
) -> SaveArtifactResponse:
    try:
        artifact_id = _repository(user_id).save_artifact(
            dataset_id=dataset_id,
            artifact_type=request.artifact_type,
            file_name=request.file_name,
            content=request.content,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SaveArtifactResponse(id=artifact_id)


@router.post("/datasets/{dataset_id}/cleaning-runs", response_model=DatasetCleaningRunResponse)
def run_cleaning_result(
    dataset_id: UUID,
    request: RunDatasetCleaningRequest,
    user_id: str = Depends(current_user_id),
) -> DatasetCleaningRunResponse:
    try:
        repository = _repository(user_id)
        raw_records = repository.read_raw_records(dataset_id)
        previous_records = repository.read_cleaned_records(dataset_id)
        cleaning_result = DataCleaningService().clean(
            dataset_id=dataset_id,
            records=raw_records,
            requirement=request.requirement,
            use_llm=request.use_llm,
        )
        raw_summary = _record_summary(raw_records)
        previous_summary = _record_summary(previous_records)
        current_summary = _record_summary(cleaning_result.records)
        diff_summary = _cleaning_diff_summary(
            raw_records=raw_records,
            previous_records=previous_records,
            current_records=cleaning_result.records,
        )
        cleaned_count = repository.save_cleaned_records(
            dataset_id=dataset_id,
            records=cleaning_result.records,
            metadata={
                "provider": cleaning_result.provider,
                "model": cleaning_result.model,
                "source": cleaning_result.source,
                "requirement": request.requirement,
                "warnings": list(cleaning_result.warnings),
            },
        )
        run_id = repository.save_cleaning_result(
            dataset_id=dataset_id,
            provider=cleaning_result.provider,
            model=cleaning_result.model,
            prompt=request.requirement or "通用分析前数据清洗",
            result_markdown=cleaning_result.result_markdown,
            cleaned_dataset={
                "status": "completed",
                "source": cleaning_result.source,
                "rows": cleaned_count,
                "columns": len(cleaning_result.records[0]) if cleaning_result.records else 0,
                "warnings": list(cleaning_result.warnings),
                "records": cleaning_result.records,
            },
            raw_summary=raw_summary,
            previous_summary=previous_summary,
            current_summary=current_summary,
            diff_summary=diff_summary,
        )
        run = repository.get_cleaning_run(dataset_id, run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DatasetCleaningRunResponse(
        dataset_id=dataset_id,
        run_id=run_id,
        version=int(run.get("version") or 1),
        provider=cleaning_result.provider,
        model=cleaning_result.model,
        source=cleaning_result.source,
        raw_row_count=len(raw_records),
        cleaned_row_count=cleaned_count,
        cleaned_column_count=len(cleaning_result.records[0]) if cleaning_result.records else 0,
        result_markdown=cleaning_result.result_markdown,
        preview_records=tuple(cleaning_result.records[:50]),
        warnings=cleaning_result.warnings,
    )


@router.get("/datasets/{dataset_id}/report-versions", response_model=ReportVersionListResponse)
def list_report_versions(
    dataset_id: UUID,
    user_id: str = Depends(current_user_id),
) -> ReportVersionListResponse:
    try:
        reports = _repository(user_id).list_report_versions(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReportVersionListResponse(
        versions=tuple(_report_version_response(report) for report in reports)
    )


@router.post("/datasets/{dataset_id}/charts", response_model=SaveArtifactResponse)
def save_chart(
    dataset_id: UUID,
    request: SaveChartRequest,
    user_id: str = Depends(current_user_id),
) -> SaveArtifactResponse:
    try:
        chart_id = _repository(user_id).save_chart(
            dataset_id=dataset_id,
            title=request.title,
            chart_type=request.chart_type,
            chart_spec=request.chart_spec,
            chart_data=request.chart_data,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SaveArtifactResponse(id=chart_id)


@router.post("/datasets/{dataset_id}/reports", response_model=SaveArtifactResponse)
def save_report(
    dataset_id: UUID,
    request: SaveReportRequest,
    user_id: str = Depends(current_user_id),
) -> SaveArtifactResponse:
    try:
        report_id = _repository(user_id).save_report(
            dataset_id=dataset_id,
            title=request.title,
            markdown=request.markdown,
            metadata=request.metadata,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SaveArtifactResponse(id=report_id)


def _repository(user_id: str = "default") -> DatasetStoreRepository:
    return DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)


def _dataset_response(dataset: StoredDataset) -> CreateDatasetResponse:
    return CreateDatasetResponse(
        dataset_id=dataset.id,
        user_id=dataset.user_id,
        name=dataset.name,
        source_type=dataset.source_type,
        status=dataset.status,
        source_metadata=dataset.source_metadata or {},
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


def _dataset_group_response(
    repository: DatasetStoreRepository,
    group: StoredDatasetGroup,
) -> DatasetGroupResponse:
    tables: list[DatasetGroupTable] = []
    for dataset_id in group.dataset_ids:
        dataset = repository.get_dataset(dataset_id)
        records = repository.read_analysis_records(dataset_id)
        profile = apply_column_metadata_to_profile(
            DatasetProfiler().profile(dataset_id=dataset_id, records=records),
            repository.list_column_metadata(dataset_id),
        )
        tables.append(
            DatasetGroupTable(
                dataset=_dataset_response(dataset),
                row_count=profile.row_count,
                column_count=profile.column_count,
                columns=tuple(column.name for column in profile.columns),
                entity_type=_infer_group_table_entity_type(
                    dataset.name, tuple(column.name for column in profile.columns)
                ),
                sample_records=tuple(records[:10]),
            )
        )
    relationships: list[DatasetRelationshipPlan] = []
    for item in group.relationships:
        try:
            relationships.append(DatasetRelationshipPlan.model_validate(item))
        except Exception:
            continue
    return DatasetGroupResponse(
        group_id=group.id,
        user_id=group.user_id,
        name=group.name,
        description=group.description,
        tables=tuple(tables),
        relationships=tuple(relationships),
        metadata=group.metadata,
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


def _infer_group_table_entity_type(name: str, columns: tuple[str, ...]) -> str:
    lowered = name.lower()
    normalized_columns = {
        "".join(character for character in column.lower() if character.isalnum())
        for column in columns
    }
    if "all_data" in lowered or ("orderid" in normalized_columns and len(columns) > 18):
        return "wide"
    if any(token in lowered for token in ("item", "line", "detail", "payment", "review")):
        return "bridge"
    if any(token in lowered for token in ("order", "sale", "transaction", "invoice")):
        return "fact"
    if any(token in lowered for token in ("translation", "lookup", "category")):
        return "lookup"
    if any(
        token in lowered for token in ("customer", "product", "seller", "geo", "region", "user")
    ):
        return "dimension"
    return "unknown"


def _report_response(
    report: dict[str, object], *, include_content: bool = True
) -> DatasetReportResponse:
    metadata = report.get("metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    if not include_content:
        metadata_payload = {
            key: metadata_payload.get(key)
            for key in (
                "question",
                "route",
                "workflow",
                "nodes",
                "planner_source",
                "sql_source",
                "python_source",
                "report_source",
                "validation_issue_count",
            )
            if key in metadata_payload
        }
    return DatasetReportResponse(
        id=UUID(str(report["id"])),
        dataset_id=UUID(str(report["dataset_id"])),
        title=str(report.get("title") or "DataMind 分析报告"),
        markdown=str(report.get("markdown") or "") if include_content else "",
        metadata=metadata_payload,
        created_at=str(report.get("created_at")) if report.get("created_at") else None,
        updated_at=str(report.get("updated_at")) if report.get("updated_at") else None,
        version=int(report.get("version") or 1),
    )


def _report_version_response(report: dict[str, object]) -> ReportVersionSummary:
    metadata = report.get("metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    return ReportVersionSummary(
        report_id=UUID(str(report["id"])),
        dataset_id=UUID(str(report["dataset_id"])),
        title=str(report.get("title") or "DataMind 分析报告"),
        question=str(metadata_payload.get("question"))
        if metadata_payload.get("question")
        else None,
        version=int(report.get("version") or 1),
        created_at=str(report.get("created_at")) if report.get("created_at") else None,
        updated_at=str(report.get("updated_at")) if report.get("updated_at") else None,
    )


def _column_response(column: dict[str, Any]) -> DatasetColumnMetadata:
    return DatasetColumnMetadata(
        column_name=str(column.get("column_name") or ""),
        inferred_type=str(column.get("inferred_type") or "text"),
        override_type=str(column.get("override_type")) if column.get("override_type") else None,
        role=str(column.get("role") or "dimension"),
        description=str(column.get("description") or ""),
        created_at=str(column.get("created_at")) if column.get("created_at") else None,
        updated_at=str(column.get("updated_at")) if column.get("updated_at") else None,
    )


def _semantic_model_response(model: dict[str, Any]) -> SemanticModelResponse:
    validation = (
        model.get("validation")
        if isinstance(model.get("validation"), dict) and model.get("validation")
        else None
    )
    return SemanticModelResponse(
        model_id=UUID(str(model["id"])),
        user_id=str(model["user_id"]),
        scope_type=str(model["scope_type"]),
        scope_id=UUID(str(model["scope_id"])),
        name=str(model["name"]),
        version=int(model["version"]),
        revision=int(model["revision"]),
        status=str(model["status"]),
        source=str(model["source"]),
        parent_model_id=UUID(str(model["parent_model_id"]))
        if model.get("parent_model_id")
        else None,
        definition=model.get("definition") or {},
        schema_fingerprint=str(model.get("schema_fingerprint") or ""),
        validation=SemanticValidationResponse.model_validate(validation) if validation else None,
        created_at=model.get("created_at"),
        updated_at=model.get("updated_at"),
        published_at=model.get("published_at"),
    )


def _cleaning_run_response(run: dict[str, Any]) -> CleaningRunDetail:
    cleaned_dataset = (
        run.get("cleaned_dataset") if isinstance(run.get("cleaned_dataset"), dict) else {}
    )
    public_cleaned_dataset = {
        key: value for key, value in cleaned_dataset.items() if key != "records"
    }
    return CleaningRunDetail(
        id=UUID(str(run["id"])),
        dataset_id=UUID(str(run["dataset_id"])),
        version=int(run.get("version") or 1),
        is_active=bool(run.get("is_active")),
        provider=str(run.get("provider") or "unknown"),
        model=str(run.get("model") or "unknown"),
        prompt=str(run.get("prompt") or ""),
        result_markdown=str(run.get("result_markdown") or ""),
        cleaned_dataset=public_cleaned_dataset,
        raw_summary=run.get("raw_summary") if isinstance(run.get("raw_summary"), dict) else {},
        previous_summary=run.get("previous_summary")
        if isinstance(run.get("previous_summary"), dict)
        else {},
        current_summary=run.get("current_summary")
        if isinstance(run.get("current_summary"), dict)
        else {},
        diff_summary=_diff_response(run.get("diff_summary")),
        created_at=str(run.get("created_at")) if run.get("created_at") else None,
    )


def _cleaning_job_response(job: Any) -> CleaningJobResponse:
    events = tuple(item for item in job.events if isinstance(item, dict))
    return CleaningJobResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        requirement=job.requirement,
        prompt_overrides=job.prompt_overrides,
        cleaning_strategy=job.cleaning_strategy,
        selected_strategy=job.selected_strategy,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        events=events,
        loop_summary=job.loop_summary or {},
        terminal_reason=job.terminal_reason,
        error=job.error,
        cleaning_run_id=job.cleaning_run_id,
        retry_of=job.retry_of,
        cancel_requested=job.cancel_requested,
        attempt=job.attempt_count,
        last_event_sequence=max((int(item.get("sequence") or 0) for item in events), default=0),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _diff_response(value: object) -> CleaningDiffSummary:
    payload = value if isinstance(value, dict) else {}
    return CleaningDiffSummary(
        raw_row_count=int(payload.get("raw_row_count") or 0),
        previous_row_count=int(payload.get("previous_row_count") or 0),
        current_row_count=int(payload.get("current_row_count") or 0),
        added_rows=int(payload.get("added_rows") or 0),
        removed_rows=int(payload.get("removed_rows") or 0),
        changed_rows=int(payload.get("changed_rows") or 0),
        added_columns=tuple(str(item) for item in payload.get("added_columns") or ()),
        removed_columns=tuple(str(item) for item in payload.get("removed_columns") or ()),
        changed_cells=int(payload.get("changed_cells") or 0),
        raw_missing_count=int(payload.get("raw_missing_count") or 0),
        previous_missing_count=int(payload.get("previous_missing_count") or 0),
        current_missing_count=int(payload.get("current_missing_count") or 0),
        sample_diffs=tuple(
            item for item in payload.get("sample_diffs") or () if isinstance(item, dict)
        ),
    )


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns: list[str] = []
    missing_count = 0
    for record in records:
        for key, value in record.items():
            column = str(key)
            if column not in columns:
                columns.append(column)
            if value is None or value == "":
                missing_count += 1
    return {
        "row_count": len(records),
        "column_count": len(columns),
        "columns": columns,
        "missing_count": missing_count,
    }


def _cleaning_diff_summary(
    *,
    raw_records: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
) -> dict[str, Any]:
    previous = previous_records or raw_records
    previous_columns = _columns_for(previous)
    current_columns = _columns_for(current_records)
    changed_rows = 0
    changed_cells = 0
    sample_diffs: list[dict[str, Any]] = []
    for index in range(min(len(previous), len(current_records))):
        before = previous[index]
        after = current_records[index]
        row_changes: dict[str, dict[str, Any]] = {}
        for column in sorted(set(before) | set(after)):
            if before.get(column) != after.get(column):
                changed_cells += 1
                row_changes[column] = {
                    "before": before.get(column),
                    "after": after.get(column),
                }
        if row_changes:
            changed_rows += 1
            if len(sample_diffs) < 20:
                sample_diffs.append(
                    {
                        "row_number": index + 1,
                        "changes": row_changes,
                    }
                )
    return {
        "raw_row_count": len(raw_records),
        "previous_row_count": len(previous),
        "current_row_count": len(current_records),
        "added_rows": max(len(current_records) - len(previous), 0),
        "removed_rows": max(len(previous) - len(current_records), 0),
        "changed_rows": changed_rows,
        "added_columns": sorted(current_columns - previous_columns),
        "removed_columns": sorted(previous_columns - current_columns),
        "changed_cells": changed_cells,
        "raw_missing_count": _record_summary(raw_records)["missing_count"],
        "previous_missing_count": _record_summary(previous)["missing_count"],
        "current_missing_count": _record_summary(current_records)["missing_count"],
        "sample_diffs": sample_diffs,
    }


def _columns_for(records: list[dict[str, Any]]) -> set[str]:
    columns: set[str] = set()
    for record in records:
        columns.update(str(key) for key in record)
    return columns


def _cleaning_rules_markdown(
    applied_rules: list[dict[str, Any]],
    issues: list[str],
) -> str:
    lines = ["# 手动清洗规则结果", ""]
    if applied_rules:
        lines.append("## 已应用规则")
        for index, rule in enumerate(applied_rules, 1):
            lines.append(f"- {index}. {rule.get('rule_type')} · {rule.get('column') or 'dataset'}")
        lines.append("")
    else:
        lines.extend(["## 已应用规则", "- 无", ""])
    if issues:
        lines.append("## 校验问题")
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")
    return "\n".join(lines)
