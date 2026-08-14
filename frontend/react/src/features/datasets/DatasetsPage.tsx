import React, { useEffect, useState } from "react";
import {
  CircleCheckBig,
  Database,
  Eye,
  Loader2,
  Network,
  PackageOpen,
  Sparkles,
  Table2,
  Trash2,
  Upload,
} from "lucide-react";
import {
  apiDelete,
  apiGet,
  apiPatch,
  apiPost,
  apiPostForm,
  runDatasetCleaning,
  type CleaningJob,
} from "../../api-client";
import { CleaningLoopPanel } from "../../cleaning-loop-ui";
import { Alert, LoadingLine, Metric } from "../../components/primitives";
import type {
  ApiState,
  Dataset,
  DatasetColumnMetadata,
  DatasetDetail,
  DatasetGroup,
  DatasetImportPipelineState,
  DatasetProfile,
  DatasetRelationshipAutoConfigureResponse,
  DatasetWorkspaceView,
  ExcelSheetPreview,
  UploadQueueItem,
} from "../../domain-types";
import {
  errorMessage,
  formatRelationship,
  formatRelationshipMetrics,
  formatTime,
  relationshipKey,
  translateStatus,
  uploadStatusLabel,
} from "../../formatters";
import { DataTable, valueText } from "../reports/ReportContent";
import { DriftMonitorPanel } from "../data-reliability/DriftMonitorPanel";
import { SemanticModelWorkbench } from "../semantic/SemanticModelWorkbench";
import {
  CleaningRuleEditor as DatasetCleaningRuleEditor,
  CleaningVersionsPanel as DatasetCleaningVersionsPanel,
  ColumnMetadataEditor as DatasetColumnMetadataEditor,
  type CleaningRule,
  type CleaningRulePreviewResponse,
  type CleaningRunDetail,
} from "./CleaningWorkspace";

export function DatasetsPage({
  datasets,
  datasetGroups,
  activeDatasetId,
  onActiveDatasetChange,
  onRefresh,
  loading,
  error,
  workspaceView,
  onWorkspaceViewChange,
  onCleaningJobUpdate,
}: {
  datasets: Dataset[];
  datasetGroups: DatasetGroup[];
  activeDatasetId: string | null;
  onActiveDatasetChange: (datasetId: string | null) => void;
  onRefresh: () => Promise<void>;
  loading: boolean;
  error: string | null;
  workspaceView: DatasetWorkspaceView;
  onWorkspaceViewChange: (view: DatasetWorkspaceView) => void;
  onCleaningJobUpdate: (job: CleaningJob) => void;
}) {
  const [requirement, setRequirement] = useState("");
  const [uploadItems, setUploadItems] = useState<UploadQueueItem[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [filePickerActive, setFilePickerActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [importPipeline, setImportPipeline] = useState<DatasetImportPipelineState>({
    phase: "idle",
    message: "",
    tableCount: 0,
    relationshipCount: 0,
    unresolvedCount: 0,
    llmUsed: false,
  });
  const [relationshipStartedAt, setRelationshipStartedAt] = useState<number | null>(null);
  const [relationshipElapsed, setRelationshipElapsed] = useState(0);
  const [preview, setPreview] = useState<Record<string, unknown>[]>([]);
  const [currentCleaningJob, setCurrentCleaningJob] = useState<CleaningJob | null>(null);
  const [detail, setDetail] = useState<ApiState<DatasetDetail>>({
    data: null,
    loading: false,
    error: null,
  });
  const groupedDatasetIds = new Set(
    datasetGroups.flatMap((group) => group.tables.map((table) => table.dataset.dataset_id)),
  );
  const ungroupedDatasets = datasets.filter((dataset) => !groupedDatasetIds.has(dataset.dataset_id));
  const selectedDataset = datasets.find((dataset) => dataset.dataset_id === activeDatasetId) ?? null;
  const pendingUploadItems = uploadItems.filter((item) => item.status === "ready");
  const cleanedDatasetCount = datasets.filter((dataset) => dataset.status === "cleaned").length;

  useEffect(() => {
    if (importPipeline.phase !== "relationships" || relationshipStartedAt === null) return;
    const updateElapsed = () => setRelationshipElapsed(Math.floor((Date.now() - relationshipStartedAt) / 1000));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [importPipeline.phase, relationshipStartedAt]);

  const updateUploadItem = (itemId: string, patch: Partial<UploadQueueItem>) => {
    setUploadItems((current) =>
      current.map((item) => (item.id === itemId ? { ...item, ...patch } : item)),
    );
  };

  const chooseFiles = async (fileList: FileList | null, append = false) => {
    setFilePickerActive(false);
    const files = Array.from(fileList ?? []);
    setMessage(null);
    if (!files.length) return;
    const nextItems: UploadQueueItem[] = files.map((nextFile, index) => {
      const suffix = nextFile.name.toLowerCase().split(".").pop() ?? "";
      const supported = ["csv", "xlsx", "json", "txt"].includes(suffix);
      return {
        id: `${Date.now()}-${index}-${nextFile.name}`,
        file: nextFile,
        sheetPreviews: [],
        selectedSheetName: "",
        status: supported ? "ready" : "error",
        message: supported ? undefined : "仅支持 CSV、XLSX、JSON、TXT。",
      };
    });
    setUploadItems((current) => (append ? [...current, ...nextItems] : nextItems));
    for (const item of nextItems) {
      if (item.status === "error" || !item.file.name.toLowerCase().endsWith(".xlsx")) continue;
      updateUploadItem(item.id, { status: "previewing", message: "正在读取 Excel sheets..." });
      try {
        const formData = new FormData();
        formData.append("file", item.file);
        const payload = await apiPostForm<{ sheets: ExcelSheetPreview[] }>("/store/files/xlsx-sheets", formData, 60000);
        updateUploadItem(item.id, {
          status: "ready",
          sheetPreviews: payload.sheets,
          selectedSheetName: payload.sheets.find((sheet) => sheet.selected)?.sheet_name ?? payload.sheets[0]?.sheet_name ?? "",
          message: "Excel sheet 已读取，请确认导入 sheet。",
        });
      } catch (err) {
        updateUploadItem(item.id, { status: "error", message: errorMessage(err) });
      }
    }
  };

  const handleUploadDrag = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (busy || filePickerActive) return;
    if (event.type === "dragenter" || event.type === "dragover") {
      setDragActive(true);
    }
    if (event.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleUploadDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    if (busy || filePickerActive) return;
    await chooseFiles(event.dataTransfer.files, true);
  };

  useEffect(() => {
    if (!filePickerActive) return;
    const resetFilePicker = () => window.setTimeout(() => setFilePickerActive(false), 250);
    window.addEventListener("focus", resetFilePicker);
    return () => window.removeEventListener("focus", resetFilePicker);
  }, [filePickerActive]);

  useEffect(() => {
    if (!activeDatasetId) {
      setDetail({ data: null, loading: false, error: null });
      return;
    }
    let isCurrent = true;
    setDetail((state) => ({ ...state, loading: true, error: null }));
    Promise.all([
      apiGet<DatasetProfile>(`/store/datasets/${activeDatasetId}/profile`),
      apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${activeDatasetId}/preview?source=raw&limit=80`),
      apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${activeDatasetId}/preview?source=cleaned&limit=80`),
      apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${activeDatasetId}/preview?source=analysis&limit=80`),
      apiGet<{ runs: CleaningRunDetail[] }>(`/store/datasets/${activeDatasetId}/cleaning-runs`),
      apiGet<{ columns: DatasetColumnMetadata[] }>(`/store/datasets/${activeDatasetId}/columns`),
    ])
      .then(([profile, rawPreview, cleanedPreview, analysisPreview, cleaningRuns, columns]) => {
        if (!isCurrent) return;
        setDetail({
          data: {
            profile,
            rawRecords: rawPreview.records,
            cleanedRecords: cleanedPreview.records,
            analysisRecords: analysisPreview.records,
            cleaningRuns: cleaningRuns.runs,
            columnMetadata: columns.columns,
          },
          loading: false,
          error: null,
        });
      })
      .catch((err) => {
        if (!isCurrent) return;
        setDetail({ data: null, loading: false, error: errorMessage(err) });
      });
    return () => {
      isCurrent = false;
    };
  }, [activeDatasetId]);

  const uploadDatasets = async () => {
    if (!uploadItems.length) {
      setMessage("请先选择一个或多个 CSV、XLSX、JSON 或 TXT 文件。");
      return;
    }
    if (!pendingUploadItems.length) {
      if (uploadItems.every((item) => item.status === "done")) {
        setMessage("这批文件已经导入并清洗完成，无需重复处理。");
        return;
      }
      if (uploadItems.every((item) => item.status === "error")) {
        setMessage("当前队列里的文件都不可处理，请移除失败项后重新选择文件。");
        return;
      }
      setMessage("当前没有待处理文件。请等待 Excel sheet 读取完成，或重新选择文件。");
      return;
    }
    setBusy(true);
    setMessage(null);
    setImportPipeline({
      phase: "processing",
      message: `正在依次导入并清洗 ${pendingUploadItems.length} 个文件。`,
      tableCount: pendingUploadItems.length,
      relationshipCount: 0,
      unresolvedCount: 0,
      llmUsed: false,
    });
    let successCount = 0;
    let failureCount = 0;
    let processedCount = 0;
    let lastDatasetId: string | null = null;
    const importedDatasetIds: string[] = [];
    try {
      for (const item of pendingUploadItems) {
        processedCount += 1;
        try {
          updateUploadItem(item.id, { status: "uploading", message: "正在导入..." });
          const formData = new FormData();
          formData.append("file", item.file);
          formData.append("dataset_name", item.file.name);
          if (item.selectedSheetName) {
            formData.append("sheet_name", item.selectedSheetName);
          }
          const payload = await apiPostForm<{
            dataset: Dataset;
            inserted: number;
            preview_records: Record<string, unknown>[];
          }>("/store/files/import", formData, 180000);
          updateUploadItem(item.id, {
            status: "cleaning",
            inserted: payload.inserted,
            datasetId: payload.dataset.dataset_id,
            message: `已导入 ${payload.inserted} 行，正在清洗...`,
          });
          const cleaning = await runDatasetCleaning(
            payload.dataset.dataset_id,
            {
              requirement,
              strategy: "auto",
              onJob: (job) => {
                setCurrentCleaningJob(job);
                onCleaningJobUpdate(job);
              },
            },
          );
          setPreview(cleaning.preview_records ?? payload.preview_records);
          updateUploadItem(item.id, {
            status: "done",
            inserted: payload.inserted,
            datasetId: payload.dataset.dataset_id,
            message: `完成：导入 ${payload.inserted} 行并创建清洗版本。`,
          });
          successCount += 1;
          lastDatasetId = payload.dataset.dataset_id;
          importedDatasetIds.push(payload.dataset.dataset_id);
        } catch (err) {
          failureCount += 1;
          updateUploadItem(item.id, { status: "error", message: errorMessage(err) });
        }
      }
      let relationshipTail = "";
      if (importedDatasetIds.length > 1) {
        try {
          setImportPipeline((current) => ({
            ...current,
            phase: "creating_group",
            message: `清洗完成，正在把 ${importedDatasetIds.length} 张表整理为一个数据包。`,
            tableCount: importedDatasetIds.length,
          }));
          const groupName = `数据包 ${new Date().toLocaleString()}`;
          const group = await apiPost<DatasetGroup>("/store/dataset-groups", {
            name: groupName,
            dataset_ids: importedDatasetIds,
            description: "由同一批拖拽/批量上传文件自动创建。",
            metadata: { source: "batch_upload", file_count: importedDatasetIds.length },
          });
          setRelationshipStartedAt(Date.now());
          setRelationshipElapsed(0);
          setImportPipeline((current) => ({
            ...current,
            phase: "relationships",
            message: "正在通过规则、压缩语义上下文和样本匹配自动识别并保存关系。",
          }));
          const configured = await apiPost<DatasetRelationshipAutoConfigureResponse>(
            `/store/dataset-groups/${group.group_id}/relationships/auto-configure`,
            {},
            120000,
          );
          const relationshipCount = configured.saved_relationships.length;
          const unresolvedCount = configured.unresolved_dataset_ids.length;
          const phase = relationshipCount ? (unresolvedCount ? "attention" : "complete") : "attention";
          setImportPipeline({
            phase,
            message: relationshipCount
              ? `已自动保存 ${relationshipCount} 条可靠关系${unresolvedCount ? `，另有 ${unresolvedCount} 张表暂未可靠关联` : "，数据包可以直接用于分析"}。`
              : "没有关系通过自动校验，数据集已保留，可在关系管理中重新识别。",
            tableCount: importedDatasetIds.length,
            relationshipCount,
            unresolvedCount,
            llmUsed: configured.llm_used,
          });
          relationshipTail = relationshipCount
            ? `已自动创建数据包并保存 ${relationshipCount} 条关系${unresolvedCount ? `，${unresolvedCount} 张表暂未关联。` : "，可直接开始分析。"}`
            : "已创建数据包，但没有关系通过自动校验。";
        } catch (err) {
          const relationshipError = errorMessage(err);
          setImportPipeline({
            phase: "error",
            message: `文件已完成导入和清洗，但自动关系识别失败：${relationshipError}`,
            tableCount: importedDatasetIds.length,
            relationshipCount: 0,
            unresolvedCount: importedDatasetIds.length,
            llmUsed: false,
          });
          relationshipTail = "文件已完成导入和清洗，但自动关系识别失败，可稍后在关系管理中重新运行。";
        } finally {
          setRelationshipStartedAt(null);
        }
      } else if (successCount > 0) {
        setImportPipeline({
          phase: "complete",
          message: "单文件已完成导入与清洗，可以直接开始分析。",
          tableCount: 1,
          relationshipCount: 0,
          unresolvedCount: 0,
          llmUsed: false,
        });
      } else {
        setImportPipeline({
          phase: "error",
          message: "没有文件成功完成导入和清洗，请检查各文件错误后重试。",
          tableCount: 0,
          relationshipCount: 0,
          unresolvedCount: 0,
          llmUsed: false,
        });
      }
      if (lastDatasetId) onActiveDatasetChange(lastDatasetId);
      await onRefresh();
      if (!processedCount) {
        setMessage("当前没有待处理文件。");
      } else {
        const tail =
          successCount === 0
            ? "没有成功完成清洗的数据集，请查看每个文件的错误信息。"
            : importedDatasetIds.length > 1
              ? relationshipTail
              : "后续分析会使用清洗后数据。";
        setMessage(`批量处理完成：本次处理 ${processedCount} 个，成功 ${successCount} 个，失败 ${failureCount} 个。${tail}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const deleteDataset = async (datasetId: string) => {
    await apiDelete(`/store/datasets/${datasetId}`);
    if (activeDatasetId === datasetId) onActiveDatasetChange(null);
    await onRefresh();
  };

  const deleteDatasetGroup = async (group: DatasetGroup) => {
    await apiDelete(`/store/dataset-groups/${group.group_id}?delete_datasets=true`);
    const deletedIds = new Set(group.tables.map((table) => table.dataset.dataset_id));
    if (activeDatasetId && deletedIds.has(activeDatasetId)) onActiveDatasetChange(null);
    await onRefresh();
  };

  const refreshDetail = async () => {
    if (!activeDatasetId) return;
    setDetail((state) => ({ ...state, loading: true, error: null }));
    try {
      const [profile, rawPreview, cleanedPreview, analysisPreview, cleaningRuns, columns] = await Promise.all([
        apiGet<DatasetProfile>(`/store/datasets/${activeDatasetId}/profile`),
        apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${activeDatasetId}/preview?source=raw&limit=80`),
        apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${activeDatasetId}/preview?source=cleaned&limit=80`),
        apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${activeDatasetId}/preview?source=analysis&limit=80`),
        apiGet<{ runs: CleaningRunDetail[] }>(`/store/datasets/${activeDatasetId}/cleaning-runs`),
        apiGet<{ columns: DatasetColumnMetadata[] }>(`/store/datasets/${activeDatasetId}/columns`),
      ]);
      setDetail({
        data: {
          profile,
          rawRecords: rawPreview.records,
          cleanedRecords: cleanedPreview.records,
          analysisRecords: analysisPreview.records,
          cleaningRuns: cleaningRuns.runs,
          columnMetadata: columns.columns,
        },
        loading: false,
        error: null,
      });
    } catch (err) {
      setDetail((state) => ({ ...state, loading: false, error: errorMessage(err) }));
    }
  };

  const updateColumnMetadata = async (columnName: string, payload: Partial<DatasetColumnMetadata>) => {
    if (!activeDatasetId) return;
    await apiPatch<DatasetColumnMetadata>(
      `/store/datasets/${activeDatasetId}/columns/${encodeURIComponent(columnName)}`,
      payload,
    );
    await refreshDetail();
  };

  const activateCleaningRun = async (runId: string) => {
    if (!activeDatasetId) return;
    await apiPost<CleaningRunDetail>(`/store/datasets/${activeDatasetId}/cleaning-runs/${runId}/activate`, {});
    await refreshDetail();
    await onRefresh();
  };

  const previewCleaningRules = async (rules: CleaningRule[]) => {
    if (!activeDatasetId) throw new Error("请先选择数据集。");
    return await apiPost<CleaningRulePreviewResponse>(
      `/store/datasets/${activeDatasetId}/cleaning-rules/preview`,
      { rules },
      60000,
    );
  };

  const applyCleaningRules = async (rules: CleaningRule[]) => {
    if (!activeDatasetId) throw new Error("请先选择数据集。");
    const result = await apiPost<{ preview_records: Record<string, unknown>[] }>(
      `/store/datasets/${activeDatasetId}/cleaning-rules/apply`,
      { rules },
      60000,
    );
    setPreview(result.preview_records ?? []);
    await refreshDetail();
    await onRefresh();
  };

  return (
    <section>
      <div className="dataset-page-header">
        <div>
          <h2 className="page-title mb-2">数据集</h2>
          <p className="max-w-2xl text-sm font-semibold leading-6 text-slate-600">
            导入数据、管理多文件关系，并在开始分析前完成清洗与字段准备。
          </p>
        </div>
        <div className="dataset-overview" aria-label="数据资产概览">
          <div>
            <Database size={17} />
            <span><b>{datasets.length}</b> 个数据集</span>
          </div>
          <div>
            <PackageOpen size={17} />
            <span><b>{datasetGroups.length}</b> 个数据包</span>
          </div>
          <div>
            <CircleCheckBig size={17} />
            <span><b>{cleanedDatasetCount}</b> 个已清洗</span>
          </div>
        </div>
      </div>
      {error && <Alert tone="error">{error}</Alert>}
      <div className="dataset-import-card">
        <div className="dataset-import-heading">
          <div className="dataset-import-icon"><Sparkles size={20} /></div>
          <div>
            <h3>导入工作台</h3>
            <p>同一批多文件会自动整合为数据包，并创建独立清洗版本。</p>
          </div>
        </div>
        <div className="dataset-import-grid">
          <div>
            <label className="label" htmlFor="dataset-cleaning-requirement">清洗需求</label>
            <textarea
              id="dataset-cleaning-requirement"
              value={requirement}
              onChange={(event) => setRequirement(event.target.value)}
              className="input min-h-[154px] resize-none"
              placeholder="例如：删除重复记录，统一日期格式，清理空白姓名，并输出可继续分析的数据集。"
            />
            <p className="mt-2 text-xs font-semibold leading-5 text-slate-500">
              批量文件会逐个导入，文件越多等待时间可能增加；多文件会优先使用本地快速清洗。
            </p>
          </div>
          <div
            className={`dataset-dropzone ${dragActive && !filePickerActive ? "is-active" : ""}`}
            onDragEnter={handleUploadDrag}
            onDragOver={handleUploadDrag}
            onDragLeave={handleUploadDrag}
            onDrop={(event) => void handleUploadDrop(event)}
          >
            <div className="dataset-dropzone-icon"><Upload size={22} /></div>
            <div>
              <p className="font-black text-slate-950">
                {uploadItems.length ? `${uploadItems.length} 个文件已加入队列` : "拖拽文件到这里"}
              </p>
              <p className="mt-1 text-xs font-semibold text-slate-500">单个文件最大 200MB · CSV、XLSX、JSON、TXT</p>
            </div>
            <label className="dataset-upload-button">
              <Upload size={16} /> 选择文件
              <input
                type="file"
                multiple
                accept=".csv,.xlsx,.json,.txt"
                className="hidden"
                onClick={() => setFilePickerActive(true)}
                onChange={(event) => void chooseFiles(event.target.files, true)}
              />
            </label>
          </div>
        </div>
        {!!uploadItems.length && (
          <section className="dataset-upload-queue">
            {uploadItems.map((item) => (
              <div key={item.id} className="dataset-upload-item">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <b className="text-slate-950">{item.file.name}</b>
                    <p className="mt-1 text-xs font-bold text-slate-500">
                      {(item.file.size / 1024).toFixed(1)}KB · {uploadStatusLabel(item.status)}
                      {item.inserted ? ` · ${item.inserted} 行` : ""}
                    </p>
                    {item.message && (
                      <p className={`mt-1 text-sm font-bold ${item.status === "error" ? "text-rose-700" : "text-slate-600"}`}>
                        {item.message}
                      </p>
                    )}
                  </div>
                  <button
                    type="button"
                    className="dataset-remove-button"
                    disabled={busy}
                    onClick={() => setUploadItems((current) => current.filter((next) => next.id !== item.id))}
                    aria-label={`移除 ${item.file.name}`}
                    title="移除文件"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                {!!item.sheetPreviews.length && (
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {item.sheetPreviews.map((sheet) => (
                      <label
                        key={`${item.id}-${sheet.sheet_name}`}
                        className={`cursor-pointer rounded-xl border p-3 ${
                          item.selectedSheetName === sheet.sheet_name ? "border-emerald-300 bg-emerald-50" : "border-line bg-slate-50"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <input
                            type="radio"
                            checked={item.selectedSheetName === sheet.sheet_name}
                            onChange={() => updateUploadItem(item.id, { selectedSheetName: sheet.sheet_name })}
                          />
                          <b>{sheet.sheet_name}</b>
                          {sheet.selected && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-black text-emerald-800">推荐</span>}
                        </div>
                        <p className="mt-1 text-xs font-bold text-slate-500">
                          {sheet.row_count} 行 · {sheet.column_count} 列
                        </p>
                        <div className="mt-3">
                          <DataTable rows={sheet.preview_records.slice(0, 3)} emptyText="暂无预览。" />
                        </div>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </section>
        )}
        {importPipeline.phase !== "idle" && (
          <DatasetImportPipeline state={importPipeline} elapsed={relationshipElapsed} />
        )}
        <CleaningLoopPanel job={currentCleaningJob} />
        <div className="dataset-import-actions">
          <p>
            {pendingUploadItems.length
              ? `${pendingUploadItems.length} 个文件等待处理`
              : uploadItems.length
                ? "当前队列没有待处理文件"
                : "选择文件后可开始导入与清洗"}
          </p>
          <button disabled={busy || !pendingUploadItems.length} onClick={uploadDatasets} className="primary-button dataset-import-submit">
            {busy ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
            {busy
              ? importPipeline.phase === "relationships"
                ? "正在自动识别关系"
                : importPipeline.phase === "creating_group"
                  ? "正在创建数据包"
                  : "正在导入与清洗"
              : "导入并创建清洗任务"}
          </button>
        </div>
        {message && <Alert tone={/失败\s*[1-9]\d*\s*个|没有成功|识别失败/.test(message) ? "error" : "info"}>{message}</Alert>}
      </div>

      {preview.length > 0 && (
        <section className="mt-8">
          <h3 className="section-heading">本次清洗预览</h3>
          <DataTable rows={preview} emptyText="暂无预览数据。" />
        </section>
      )}

      <section className="dataset-workspace">
        <div className="dataset-workspace-heading">
          <div>
            <h3 className="section-heading mb-1">数据资产</h3>
            <p>在资产列表、数据包关系和当前数据详情之间切换。</p>
          </div>
          <div className="dataset-workspace-tabs" role="tablist" aria-label="数据资产视图">
            <button type="button" role="tab" aria-selected={workspaceView === "assets"} className={workspaceView === "assets" ? "is-active" : ""} onClick={() => onWorkspaceViewChange("assets")}>
              <Table2 size={16} /> 资产列表 <span>{datasets.length + datasetGroups.length}</span>
            </button>
            <button type="button" role="tab" aria-selected={workspaceView === "relationships"} className={workspaceView === "relationships" ? "is-active" : ""} onClick={() => onWorkspaceViewChange("relationships")}>
              <Network size={16} /> 关系管理 <span>{datasetGroups.length}</span>
            </button>
            <button type="button" role="tab" aria-selected={workspaceView === "detail"} className={workspaceView === "detail" ? "is-active" : ""} onClick={() => onWorkspaceViewChange("detail")}>
              <Database size={16} /> 数据详情
            </button>
          </div>
        </div>

        {workspaceView === "assets" && (
          <DatasetCollectionList
            title="已导入资产"
            groups={datasetGroups}
            datasets={ungroupedDatasets}
            loading={loading}
            onOpen={(datasetId) => {
              onActiveDatasetChange(datasetId);
              onWorkspaceViewChange("detail");
            }}
            onDelete={deleteDataset}
            onOpenGroup={(group) => {
              const firstDatasetId = group.relationships.find((relationship) => relationship.enabled !== false)?.left_dataset_id ?? group.tables[0]?.dataset.dataset_id;
              if (firstDatasetId) {
                onActiveDatasetChange(firstDatasetId);
                onWorkspaceViewChange("detail");
              }
            }}
            onDeleteGroup={deleteDatasetGroup}
          />
        )}
        {workspaceView === "relationships" && <DatasetGroupList groups={datasetGroups} onRefresh={onRefresh} />}
        {workspaceView === "detail" && (
          <DatasetDetailPanel
            dataset={selectedDataset}
            detail={detail.data}
            loading={detail.loading}
            error={detail.error}
            onUpdateColumn={updateColumnMetadata}
            onActivateCleaningRun={activateCleaningRun}
            onPreviewCleaningRules={previewCleaningRules}
            onApplyCleaningRules={applyCleaningRules}
          />
        )}
      </section>
    </section>
  );
}

function DatasetCollectionList({
  title,
  groups,
  datasets,
  loading,
  onOpen,
  onDelete,
  onOpenGroup,
  onDeleteGroup,
}: {
  title: string;
  groups: DatasetGroup[];
  datasets: Dataset[];
  loading: boolean;
  onOpen: (datasetId: string) => void;
  onDelete: (datasetId: string) => void;
  onOpenGroup: (group: DatasetGroup) => void;
  onDeleteGroup: (group: DatasetGroup) => Promise<void>;
}) {
  const [deletingDatasetId, setDeletingDatasetId] = useState<string | null>(null);
  const [deletingGroupId, setDeletingGroupId] = useState<string | null>(null);
  const hasRows = groups.length > 0 || datasets.length > 0;
  const handleDeleteDataset = async (datasetId: string) => {
    setDeletingDatasetId(datasetId);
    try {
      await onDelete(datasetId);
    } finally {
      setDeletingDatasetId(null);
    }
  };
  const handleDeleteGroup = async (group: DatasetGroup) => {
    setDeletingGroupId(group.group_id);
    try {
      await onDeleteGroup(group);
    } finally {
      setDeletingGroupId(null);
    }
  };
  return (
    <section className="dataset-asset-list">
      <div className="dataset-list-heading">
        <div>
          <h4>{title}</h4>
          <p>{groups.length} 个数据包 · {datasets.length} 个独立数据集</p>
        </div>
      </div>
      {loading && <LoadingLine />}
      {!loading && !hasRows && <Alert>还没有导入的数据集。</Alert>}
      {hasRows && (
        <div className="space-y-3">
          <div className="dataset-table-header">
            <span>名称</span>
            <span>类型</span>
            <span>状态</span>
            <span>创建时间</span>
            <span>操作</span>
          </div>
          {groups.map((group) => {
            const totalRows = group.tables.reduce((sum, table) => sum + table.row_count, 0);
            const allCleaned = group.tables.every((table) => table.dataset.status === "cleaned");
            const someCleaned = group.tables.some((table) => table.dataset.status === "cleaned");
            return (
              <div key={`group-${group.group_id}`} className="dataset-row">
                <div className="dataset-primary">
                  <div className="dataset-record-icon is-package"><PackageOpen size={18} /></div>
                  <div>
                    <b>{group.name}</b>
                    <small className="mt-1 block text-slate-500">
                      ID: {group.group_id.slice(0, 8)} · {group.tables.length} 张表 · {totalRows.toLocaleString()} 行
                    </small>
                    <small className="mt-1 block max-w-xl truncate text-slate-500">
                      {group.tables.slice(0, 4).map((table) => table.dataset.name).join(", ")}
                      {group.tables.length > 4 ? ` 等 ${group.tables.length} 个文件` : ""}
                    </small>
                  </div>
                </div>
                <span className="dataset-kind">数据包</span>
                <span className={`dataset-status ${allCleaned ? "is-cleaned" : someCleaned ? "is-partial" : ""}`}>
                  {allCleaned ? "已清洗" : someCleaned ? "部分清洗" : "已导入"}
                </span>
                <span className="dataset-created">{formatTime(group.created_at)}</span>
                <div className="dataset-row-actions">
                  <button className="dataset-icon-button" disabled={deletingGroupId === group.group_id} onClick={() => onOpenGroup(group)} aria-label={`查看 ${group.name}`} title="查看数据包">
                    <Eye size={17} />
                  </button>
                  <button className="dataset-icon-button is-danger" disabled={deletingGroupId === group.group_id} onClick={() => void handleDeleteGroup(group)} aria-label={`删除 ${group.name}`} title={deletingGroupId === group.group_id ? "删除中" : "删除数据包及其数据集"}>
                    {deletingGroupId === group.group_id ? <Loader2 className="animate-spin" size={17} /> : <Trash2 size={17} />}
                  </button>
                </div>
              </div>
            );
          })}
          {datasets.map((dataset) => (
            <div key={`dataset-${dataset.dataset_id}`} className="dataset-row">
              <div className="dataset-primary">
                <div className="dataset-record-icon"><Table2 size={18} /></div>
                <div className="min-w-0">
                  <b className="block truncate">{dataset.name}</b>
                  <small className="mt-1 block text-slate-500">ID: {dataset.dataset_id.slice(0, 8)}</small>
                </div>
              </div>
              <span className="dataset-kind">{dataset.source_type.toUpperCase()}</span>
              <span className={`dataset-status ${dataset.status === "cleaned" ? "is-cleaned" : ""}`}>
                {translateStatus(dataset.status)}
              </span>
              <span className="dataset-created">{formatTime(dataset.created_at)}</span>
              <div className="dataset-row-actions">
                <button className="dataset-icon-button" disabled={deletingDatasetId === dataset.dataset_id} onClick={() => onOpen(dataset.dataset_id)} aria-label={`查看 ${dataset.name}`} title="查看数据集">
                  <Eye size={17} />
                </button>
                <button className="dataset-icon-button is-danger" disabled={deletingDatasetId === dataset.dataset_id} onClick={() => void handleDeleteDataset(dataset.dataset_id)} aria-label={`删除 ${dataset.name}`} title={deletingDatasetId === dataset.dataset_id ? "删除中" : "删除数据集"}>
                  {deletingDatasetId === dataset.dataset_id ? <Loader2 className="animate-spin" size={17} /> : <Trash2 size={17} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function DatasetImportPipeline({ state, elapsed }: { state: DatasetImportPipelineState; elapsed: number }) {
  const steps = [
    { key: "processing", label: "导入并清洗" },
    { key: "creating_group", label: state.tableCount > 1 ? "创建数据包" : "准备分析数据" },
    { key: "relationships", label: state.tableCount > 1 ? "自动识别并保存关系" : "完成" },
  ] as const;
  const phaseOrder = { processing: 0, creating_group: 1, relationships: 2, complete: 3, attention: 3, error: 3, idle: -1 };
  const currentIndex = phaseOrder[state.phase];
  const isTerminal = ["complete", "attention", "error"].includes(state.phase);
  return (
    <section className={`dataset-import-pipeline is-${state.phase}`} role="status" aria-live="polite">
      <div className="dataset-import-pipeline-heading">
        <span>
          {state.phase === "complete" ? <CircleCheckBig size={20} /> : state.phase === "relationships" ? <Network size={20} /> : <Loader2 className={isTerminal ? "" : "animate-spin"} size={20} />}
        </span>
        <div>
          <b>
            {state.phase === "complete"
              ? "数据已准备完成"
              : state.phase === "attention"
                ? "数据已导入，部分关系需要留意"
                : state.phase === "error"
                  ? "自动处理未全部完成"
                  : state.phase === "relationships"
                    ? "正在自动建立数据关系"
                    : "正在准备数据"}
          </b>
          <p>{state.message}{state.phase === "relationships" ? ` 已等待 ${elapsed} 秒。` : ""}</p>
        </div>
      </div>
      <div className="dataset-import-pipeline-steps">
        {steps.map((step, index) => {
          const completed = isTerminal || index < currentIndex;
          const active = !isTerminal && index === currentIndex;
          return (
            <div key={step.key} className={completed ? "is-complete" : active ? "is-active" : "is-waiting"}>
              <span>{completed ? <CircleCheckBig size={15} /> : active ? <Loader2 className="animate-spin" size={15} /> : "○"}</span>
              <b>{step.label}</b>
            </div>
          );
        })}
      </div>
      {state.phase === "relationships" && (
        <>
          <div className="relationship-loading-track"><i /></div>
          <p className="dataset-import-pipeline-note">只发送压缩字段摘要和少量样本；表较多或字段命名不典型时，语义校验可能需要更长时间。</p>
        </>
      )}
      {isTerminal && state.tableCount > 1 && (
        <div className="dataset-import-pipeline-result">
          <span>{state.tableCount} 张表</span>
          <span>{state.relationshipCount} 条自动关系</span>
          {state.unresolvedCount > 0 && <span>{state.unresolvedCount} 张未关联</span>}
          {state.llmUsed && <span>已使用语义补充</span>}
        </div>
      )}
    </section>
  );
}

function DatasetGroupList({ groups, onRefresh }: { groups: DatasetGroup[]; onRefresh: () => Promise<void> }) {
  const [suggestionsByGroup, setSuggestionsByGroup] = useState<Record<string, DatasetRelationshipAutoConfigureResponse>>({});
  const [busyGroupId, setBusyGroupId] = useState<string | null>(null);
  const [messageByGroup, setMessageByGroup] = useState<Record<string, string>>({});
  const [recommendationStartedAt, setRecommendationStartedAt] = useState<number | null>(null);
  const [recommendationElapsed, setRecommendationElapsed] = useState(0);

  useEffect(() => {
    if (!busyGroupId || recommendationStartedAt === null) return;
    const updateElapsed = () => setRecommendationElapsed(Math.floor((Date.now() - recommendationStartedAt) / 1000));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [busyGroupId, recommendationStartedAt]);

  const refreshSuggestions = async (groupId: string) => {
    setBusyGroupId(groupId);
    setRecommendationStartedAt(Date.now());
    setRecommendationElapsed(0);
    setMessageByGroup((current) => ({ ...current, [groupId]: "" }));
    try {
      const payload = await apiPost<DatasetRelationshipAutoConfigureResponse>(
        `/store/dataset-groups/${groupId}/relationships/auto-configure`,
        {},
        120000,
      );
      setSuggestionsByGroup((current) => ({ ...current, [groupId]: payload }));
      setMessageByGroup((current) => ({
        ...current,
        [groupId]: payload.saved_relationships.length
          ? `已自动校验并保存 ${payload.saved_relationships.length} 条关系${payload.unresolved_dataset_ids.length ? `，${payload.unresolved_dataset_ids.length} 张表暂未关联。` : "。"}`
          : "没有候选关系通过自动校验，原始数据不受影响。",
      }));
      await onRefresh();
    } catch (err) {
      setMessageByGroup((current) => ({ ...current, [groupId]: errorMessage(err) }));
    } finally {
      setBusyGroupId(null);
      setRecommendationStartedAt(null);
    }
  };

  if (!groups.length) return null;
  return (
    <section className="mt-8">
      <h3 className="section-heading">数据包 / 多文件组</h3>
      <div className="grid gap-4">
        {groups.map((group) => {
          const hydratedGroup = suggestionsByGroup[group.group_id]?.group ?? group;
          const suggestions = suggestionsByGroup[group.group_id];
          const confirmedRelationships = hydratedGroup.relationships.filter((relationship) => relationship.enabled !== false);
          const isBusy = busyGroupId === group.group_id;
          return (
            <div key={group.group_id} className="surface-card min-w-0">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h4 className="text-lg font-black text-slate-950">{hydratedGroup.name}</h4>
                  <p className="mt-1 text-sm font-semibold text-slate-500">
                    {hydratedGroup.tables.length} 张表 · {hydratedGroup.description || "多文件批次"}
                  </p>
                </div>
                <button
                  type="button"
                  className="small-button"
                  disabled={isBusy}
                  onClick={() => void refreshSuggestions(group.group_id)}
                >
                  {isBusy && <Loader2 className="animate-spin" size={16} />}
                  {isBusy ? "正在自动配置" : suggestions || confirmedRelationships.length ? "重新识别关系" : "自动识别关系"}
                </button>
              </div>
              {isBusy ? (
                <div className="relationship-loading-panel" role="status" aria-live="polite">
                  <div className="relationship-loading-heading">
                    <span><Loader2 className="animate-spin" size={19} /></span>
                    <div>
                      <b>正在识别并保存可靠关系</b>
                      <p>已等待 {recommendationElapsed} 秒 · 系统会自动筛选字段、校验样本并持久化可执行关系</p>
                    </div>
                  </div>
                  <div className="relationship-loading-track"><i /></div>
                  <div className="relationship-loading-stages" aria-label="关系推荐处理内容">
                    <span>规则候选</span><span>语义补充</span><span>样本匹配校验</span>
                  </div>
                </div>
              ) : confirmedRelationships.length ? (
                <div className="relationship-readiness is-ready">
                  <CircleCheckBig size={18} />
                  <div><b>数据包已就绪</b><p>系统已自动保存 {confirmedRelationships.length} 条关系，可以返回分析页运行。</p></div>
                </div>
              ) : (
                <div className="relationship-readiness is-blocked">
                  <Network size={18} />
                  <div><b>尚未建立可靠关系</b><p>点击“自动识别关系”，系统会完成推荐、校验与保存。</p></div>
                </div>
              )}
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {hydratedGroup.tables.map((table) => (
                  <div key={table.dataset.dataset_id} className="rounded-xl border border-line bg-slate-50 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <b>{table.dataset.name}</b>
                      <span className="rounded-full bg-white px-2 py-1 text-xs font-black text-slate-600">{table.entity_type}</span>
                    </div>
                    <p className="mt-1 text-xs font-bold text-slate-500">
                      {table.row_count} 行 · {table.column_count} 列
                    </p>
                    <p className="mt-2 line-clamp-2 text-xs font-semibold text-slate-600">{table.columns.slice(0, 12).join(", ")}</p>
                  </div>
                ))}
              </div>
              {!!confirmedRelationships.length && (
                <div className="mt-4 grid gap-2">
                  {confirmedRelationships.map((relationship) => (
                    <div
                      key={relationshipKey(relationship)}
                      className={`rounded-lg border px-3 py-2 text-sm font-bold ${
                        relationship.freshness_status === "stale"
                          ? "border-rose-200 bg-rose-50 text-rose-900"
                          : "border-emerald-200 bg-emerald-50 text-emerald-950"
                      }`}
                    >
                      {formatRelationship(relationship, group.tables)}
                      {relationship.freshness_status === "stale"
                        ? ` · 已失效：${relationship.stale_reason || "匹配率或字段发生变化"}`
                        : ` · ${formatRelationshipMetrics(relationship)}`}
                    </div>
                  ))}
                </div>
              )}
              {messageByGroup[group.group_id] && (
                <Alert tone={/接口|错误|失败/.test(messageByGroup[group.group_id]) ? "error" : "info"}>{messageByGroup[group.group_id]}</Alert>
              )}
              {!!suggestions?.validation_issues.length && (
                <div className="mt-3 grid gap-2">
                  {suggestions.validation_issues.map((issue) => (
                    <Alert key={issue} tone="error">{issue}</Alert>
                  ))}
                </div>
              )}
              {!!suggestions?.candidates.length && (
                <div className="mt-4 grid gap-3">
                  {suggestions.candidates.slice(0, 8).map((candidate) => (
                    <div key={relationshipKey(candidate)} className="rounded-xl border border-line bg-white p-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <b>{formatRelationship(candidate, group.tables)}</b>
                          <p className="mt-1 text-xs font-bold text-slate-500">
                            {candidate.source} · 推荐置信度 {(candidate.confidence * 100).toFixed(0)}% · 样本匹配率 {(candidate.estimated_match_rate * 100).toFixed(0)}% · {candidate.relationship_type ?? "unknown"}
                          </p>
                          <p className="mt-1 text-sm font-semibold text-slate-600">{candidate.reason}</p>
                          {candidate.risk_note && <p className="mt-1 text-xs font-black text-amber-700">{candidate.risk_note}</p>}
                        </div>
                        {confirmedRelationships.some((relationship) => relationshipKey(relationship) === relationshipKey(candidate)) ? (
                          <span className="relationship-decision is-adopted"><CircleCheckBig size={15} /> 已自动采用</span>
                        ) : (
                          <span className="relationship-decision">未采用</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <DriftMonitorPanel
                groupId={hydratedGroup.group_id}
                datasetNames={Object.fromEntries(
                  hydratedGroup.tables.map((table) => [
                    table.dataset.dataset_id,
                    table.dataset.name,
                  ]),
                )}
                onScanned={onRefresh}
              />
              <SemanticModelWorkbench group={hydratedGroup} />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function DatasetDetailPanel({
  dataset,
  detail,
  loading,
  error,
  onUpdateColumn,
  onActivateCleaningRun,
  onPreviewCleaningRules,
  onApplyCleaningRules,
}: {
  dataset: Dataset | null;
  detail: DatasetDetail | null;
  loading: boolean;
  error: string | null;
  onUpdateColumn: (columnName: string, payload: Partial<DatasetColumnMetadata>) => Promise<void>;
  onActivateCleaningRun: (runId: string) => Promise<void>;
  onPreviewCleaningRules: (rules: CleaningRule[]) => Promise<CleaningRulePreviewResponse>;
  onApplyCleaningRules: (rules: CleaningRule[]) => Promise<void>;
}) {
  if (!dataset) {
    return (
      <section className="mt-8">
        <h3 className="section-heading">数据集详情</h3>
        <Alert>点击任意数据集的“查看”，这里会显示原始数据、清洗后数据、字段画像和数据健康度。</Alert>
      </section>
    );
  }

  return (
    <section className="mt-8">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h3 className="section-heading mb-2">数据集详情</h3>
          <p className="max-w-3xl text-sm font-semibold text-slate-600">{dataset.name}</p>
        </div>
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-black text-emerald-900">
          {translateStatus(dataset.status)}
        </span>
      </div>
      {loading && <LoadingLine />}
      {error && <Alert tone="error">{error}</Alert>}
      {!loading && detail && (
        <div className="space-y-6">
          <DatasetProfileSummary profile={detail.profile} dataset={dataset} />
          <DatasetCleaningVersionsPanel
            runs={detail.cleaningRuns}
            onActivate={onActivateCleaningRun}
          />
          <DatasetCleaningRuleEditor
            columns={detail.profile?.columns.map((column) => column.name) ?? Object.keys(detail.analysisRecords[0] ?? {})}
            onPreview={onPreviewCleaningRules}
            onApply={onApplyCleaningRules}
          />
          <DatasetColumnMetadataEditor
            profile={detail.profile}
            metadata={detail.columnMetadata}
            onUpdateColumn={onUpdateColumn}
          />
          <section className="surface-card">
            <h4 className="section-heading">字段画像</h4>
            <DataTable
              rows={(detail.profile?.columns ?? []).map((column) => ({
                字段: column.name,
                类型: column.dtype,
                分类: column.is_numeric ? "数值" : "类别/文本",
                缺失值: column.missing_count,
                唯一值: column.distinct_count,
                最小值: valueText(column.min_value),
                最大值: valueText(column.max_value),
                平均值: valueText(column.mean),
              }))}
              emptyText="暂无字段画像。"
            />
          </section>
          <div className="grid gap-6 xl:grid-cols-2">
            <section className="surface-card">
              <h4 className="section-heading">导入后的原始数据</h4>
              <DataTable rows={detail.rawRecords} emptyText="当前数据集没有可预览的原始行数据。" />
            </section>
            <section className="surface-card">
              <h4 className="section-heading">清洗后的数据</h4>
              <DataTable rows={detail.cleanedRecords} emptyText="当前数据集还没有清洗后数据。" />
            </section>
          </div>
          <section className="surface-card">
            <h4 className="section-heading">当前分析使用的数据</h4>
            <p className="mb-4 text-sm text-slate-500">如果存在清洗数据，分析会优先使用清洗后的数据；否则使用原始数据。</p>
            <DataTable rows={detail.analysisRecords} emptyText="当前没有可用于分析的数据。" />
          </section>
        </div>
      )}
    </section>
  );
}

function DatasetProfileSummary({ profile, dataset }: { profile: DatasetProfile | null; dataset: Dataset }) {
  const missingRatio = profile ? `${(profile.missing_value_ratio * 100).toFixed(1)}%` : "-";
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
      <Metric label="行数" value={profile?.row_count ?? "-"} caption="可分析记录数" />
      <Metric label="列数" value={profile?.column_count ?? "-"} caption="字段数量" />
      <Metric label="缺失率" value={missingRatio} caption={`${profile?.missing_value_count ?? 0} 个缺失值`} />
      <Metric label="重复行" value={profile?.duplicate_row_count ?? "-"} caption="导入后检测" />
      <Metric label="来源" value={dataset.source_type.toUpperCase()} caption={`创建于 ${formatTime(dataset.created_at)}`} />
    </div>
  );
}
