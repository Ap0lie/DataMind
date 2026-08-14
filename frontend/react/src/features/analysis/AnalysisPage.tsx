import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  CircleCheckBig,
  FileText,
  History,
  Loader2,
  MessageSquareText,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Plus,
  Search,
  Sparkles,
  SquarePen,
} from "lucide-react";
import { AgentLoopPanel } from "../../agent-loop-ui";
import { apiGet, apiPost } from "../../api-client";
import { Alert, LoadingLine } from "../../components/primitives";
import type {
  AnalysisJob,
  AnalysisResponse,
  Dataset,
  DatasetGroup,
  DatasetJoinConfig,
  DatasetProfile,
  DatasetReference,
  JoinSuggestionResponse,
  MultiDatasetContext,
  MultimodalInput,
  PlannerDecision,
  PlannerMetadata,
  PythonCodeAttempt,
  WorkflowTraceNode,
} from "../../domain-types";
import {
  errorMessage,
  formatRelationship,
  formatTime,
  relationshipKey,
  relationshipMatchRate,
} from "../../formatters";
import { isActiveAnalysisJob, jobStageLabel, jobStatusLabel } from "../../workflow-ui";
import {
  ChartList,
  DataTable,
  MetricPill,
  MultimodalContextPanel,
  StructuredReportPreview,
  TextAnalysisPanel,
  arrayOfRecords,
  structuredReportFromUnknown,
} from "../reports/ReportContent";
import { AnalysisReliabilityPanel } from "./AnalysisReliabilityPanel";
import { AnalysisJobStatusPanel, DynamicAgentPlan, RealtimeWorkflowPanel } from "./WorkflowStatus";
import { analysisErrorMessage, pollAnalysisJob } from "./job-client";

export function AnalysisPage({
  datasets,
  datasetGroups,
  activeDatasetId,
  onActiveDatasetChange,
  onReportsChanged,
  jobs,
  onJobsChanged,
  latestResult,
  onLatestResultChange,
  selectedJobId,
  onSelectedJobIdChange,
  onJobUpdate,
  onOpenDatasetRelationships,
}: {
  datasets: Dataset[];
  datasetGroups: DatasetGroup[];
  activeDatasetId: string | null;
  onActiveDatasetChange: (datasetId: string) => void;
  onReportsChanged: () => Promise<void>;
  jobs: AnalysisJob[];
  onJobsChanged: () => Promise<void>;
  latestResult: AnalysisResponse | null;
  onLatestResultChange: (result: AnalysisResponse | null) => void;
  selectedJobId: string | null;
  onSelectedJobIdChange: (jobId: string | null) => void;
  onJobUpdate: (job: AnalysisJob) => void;
  onOpenDatasetRelationships: () => void;
}) {
  const [question, setQuestion] = useState("");
  const [multimodalContext, setMultimodalContext] = useState("");
  const [multimodalFile, setMultimodalFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [currentJob, setCurrentJob] = useState<AnalysisJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedDatasetGroupId, setSelectedDatasetGroupId] = useState("");
  const [selectedAdditionalDatasetIds, setSelectedAdditionalDatasetIds] = useState<string[]>([]);
  const [joinSuggestions, setJoinSuggestions] = useState<JoinSuggestionResponse | null>(null);
  const [joinPlan, setJoinPlan] = useState<DatasetJoinConfig[]>([]);
  const [joinReferences, setJoinReferences] = useState<Record<string, DatasetReference>>({});
  const [joinBusy, setJoinBusy] = useState(false);
  const [joinError, setJoinError] = useState<string | null>(null);
  const [plannerDecision, setPlannerDecision] = useState<PlannerDecision | null>(null);
  const [confirmLowConfidence, setConfirmLowConfidence] = useState(false);
  const [agentLoopEnabled, setAgentLoopEnabled] = useState(
    () => window.localStorage.getItem("datamind.agentLoopMode.v2") !== "legacy",
  );
  const [historyCollapsed, setHistoryCollapsed] = useState(
    () => window.localStorage.getItem("datamind.analysisHistoryCollapsed.v1") === "true",
  );
  const updateCurrentJob = (job: AnalysisJob) => {
    setCurrentJob(job);
    onJobUpdate(job);
  };

  useEffect(() => {
    window.localStorage.setItem("datamind.analysisHistoryCollapsed.v1", String(historyCollapsed));
  }, [historyCollapsed]);
  useEffect(() => {
    window.localStorage.setItem("datamind.agentLoopMode.v2", agentLoopEnabled ? "loop" : "legacy");
  }, [agentLoopEnabled]);
  const selectedDatasetGroup = datasetGroups.find((group) => group.group_id === selectedDatasetGroupId) ?? null;
  const selectedGroupRelationships = (selectedDatasetGroup?.relationships ?? []).filter((relationship) => relationship.enabled !== false);
  const relationshipRightDatasetIds = new Set(selectedGroupRelationships.map((relationship) => relationship.right_dataset_id));
  const groupPrimaryDatasetId =
    selectedGroupRelationships.find((relationship) => !relationshipRightDatasetIds.has(relationship.left_dataset_id))?.left_dataset_id
    ?? selectedGroupRelationships[0]?.left_dataset_id
    ?? null;
  const selectedDatasetId = groupPrimaryDatasetId ?? activeDatasetId ?? datasets[0]?.dataset_id ?? "";
  const relationshipBlocked = !!selectedDatasetGroup && selectedGroupRelationships.length === 0;
  const groupJoinPlan: DatasetJoinConfig[] = selectedGroupRelationships.map((relationship) => ({
    left_dataset_id: relationship.left_dataset_id,
    right_dataset_id: relationship.right_dataset_id,
    left_column: relationship.left_column,
    right_column: relationship.right_column,
    join_type: relationship.join_type,
    left_value_mode: relationship.left_value_mode,
    right_value_mode: relationship.right_value_mode,
    left_delimiter: relationship.left_delimiter,
    right_delimiter: relationship.right_delimiter,
  }));
  const groupAdditionalDatasetIds = Array.from(
    new Set(
      groupJoinPlan
        .flatMap((relationship) => [relationship.left_dataset_id, relationship.right_dataset_id])
        .filter((datasetId) => datasetId !== selectedDatasetId),
    ),
  );
  const additionalDatasetOptions = datasets.filter((dataset) => dataset.dataset_id !== selectedDatasetId);

  useEffect(() => {
    if (groupPrimaryDatasetId && activeDatasetId !== groupPrimaryDatasetId) {
      onActiveDatasetChange(groupPrimaryDatasetId);
    }
  }, [groupPrimaryDatasetId, activeDatasetId]);

  useEffect(() => {
    if (!selectedJobId) {
      setCurrentJob(null);
      setError(null);
      return;
    }
    let canceled = false;
    let timer: number | null = null;
    const refreshSelectedJob = async () => {
      try {
        const job = await apiGet<AnalysisJob>(`/analysis/jobs/${selectedJobId}`);
        if (canceled) return;
        updateCurrentJob(job);
        setQuestion(job.question);
        onActiveDatasetChange(job.dataset_id);
        if (job.status === "completed") {
          const payload = await apiGet<AnalysisResponse>(`/analysis/jobs/${job.job_id}/result`);
          if (!canceled) onLatestResultChange(payload);
          return;
        }
        onLatestResultChange(null);
        if (isActiveAnalysisJob(job) && !busy) {
          timer = window.setTimeout(() => void refreshSelectedJob(), 1000);
        }
      } catch (err) {
        if (!canceled) setError(errorMessage(err));
      }
    };
    void refreshSelectedJob();
    return () => {
      canceled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [selectedJobId, busy]);

  useEffect(() => {
    setSelectedAdditionalDatasetIds((current) => current.filter((datasetId) => datasetId !== selectedDatasetId));
  }, [selectedDatasetId]);

  const selectDatasetGroup = (groupId: string) => {
    setSelectedDatasetGroupId(groupId);
    const group = datasetGroups.find((item) => item.group_id === groupId);
    if (!group) return;
    const primaryId = group.relationships.find((relationship) => relationship.enabled !== false)?.left_dataset_id ?? group.tables[0]?.dataset.dataset_id;
    if (primaryId) onActiveDatasetChange(primaryId);
    setSelectedAdditionalDatasetIds([]);
    setJoinPlan([]);
    setJoinSuggestions(null);
  };

  useEffect(() => {
    if (!selectedDatasetId || !selectedAdditionalDatasetIds.length) {
      setJoinSuggestions(null);
      setJoinPlan([]);
      setJoinReferences({});
      setJoinError(null);
      return;
    }
    let canceled = false;
    setJoinBusy(true);
    setJoinError(null);
    const selectedIds = [selectedDatasetId, ...selectedAdditionalDatasetIds];
    const loadReferences = Promise.all(
      selectedIds.map(async (datasetId) => {
        const [dataset, profile] = await Promise.all([
          Promise.resolve(datasets.find((item) => item.dataset_id === datasetId) ?? null),
          apiGet<DatasetProfile>(`/store/datasets/${datasetId}/profile`),
        ]);
        return {
          dataset_id: datasetId,
          name: dataset?.name ?? datasetId,
          status: dataset?.status ?? "",
          row_count: profile.row_count,
          column_count: profile.column_count,
          columns: profile.columns.map((column) => column.name),
        } satisfies DatasetReference;
      }),
    );
    void loadReferences
      .then((refs) => {
        if (canceled) return;
        const nextReferences = Object.fromEntries(refs.map((ref) => [ref.dataset_id, ref]));
        setJoinReferences(nextReferences);
        setJoinPlan((current) => {
          const byRight = new Map(current.map((item) => [item.right_dataset_id, item]));
          const primaryColumns = nextReferences[selectedDatasetId]?.columns ?? [];
          return selectedAdditionalDatasetIds.map((rightDatasetId) => {
            const existing = byRight.get(rightDatasetId);
            if (existing && existing.left_column && existing.right_column) return existing;
            return {
              left_dataset_id: selectedDatasetId,
              right_dataset_id: rightDatasetId,
              left_column: existing?.left_column || primaryColumns[0] || "",
              right_column: existing?.right_column || nextReferences[rightDatasetId]?.columns[0] || "",
              join_type: existing?.join_type ?? "left",
            };
          });
        });
      })
      .catch((err) => {
        if (!canceled) setJoinError(errorMessage(err));
      });
    void apiPost<JoinSuggestionResponse>(
      "/analysis/join-suggestions",
      {
        dataset_id: selectedDatasetId,
        additional_dataset_ids: selectedAdditionalDatasetIds,
      },
      90000,
    )
      .then((payload) => {
        if (canceled) return;
        setJoinSuggestions(payload);
        setJoinReferences((current) => {
          const next = { ...current, [payload.primary_dataset.dataset_id]: payload.primary_dataset };
          payload.additional_datasets.forEach((dataset) => {
            next[dataset.dataset_id] = dataset;
          });
          return next;
        });
        setJoinPlan((current) => {
          const byRight = new Map(current.map((item) => [item.right_dataset_id, item]));
          return selectedAdditionalDatasetIds.map((rightDatasetId) => {
            const existing = byRight.get(rightDatasetId);
            if (existing) return existing;
            const suggestion = payload.suggestions
              .filter((item) => item.right_dataset_id === rightDatasetId)
              .sort((a, b) => b.score - a.score)[0];
            return suggestion
              ? {
                  left_dataset_id: suggestion.left_dataset_id,
                  right_dataset_id: suggestion.right_dataset_id,
                  left_column: suggestion.left_column,
                  right_column: suggestion.right_column,
                  join_type: "left",
                  left_value_mode: suggestion.left_value_mode,
                  right_value_mode: suggestion.right_value_mode,
                  left_delimiter: suggestion.left_delimiter,
                  right_delimiter: suggestion.right_delimiter,
                }
              : {
                  left_dataset_id: selectedDatasetId,
                  right_dataset_id: rightDatasetId,
                  left_column: payload.primary_dataset.columns[0] ?? "",
                  right_column: payload.additional_datasets.find((dataset) => dataset.dataset_id === rightDatasetId)?.columns[0] ?? "",
                  join_type: "left",
                };
          });
        });
      })
      .catch((err) => {
        if (!canceled) setJoinError(errorMessage(err));
      })
      .finally(() => {
        if (!canceled) setJoinBusy(false);
      });
    return () => {
      canceled = true;
    };
  }, [selectedDatasetId, selectedAdditionalDatasetIds.join("|")]);

  const toggleAdditionalDataset = (datasetId: string) => {
    setSelectedAdditionalDatasetIds((current) =>
      current.includes(datasetId)
        ? current.filter((item) => item !== datasetId)
        : [...current, datasetId],
    );
  };

  const updateJoinConfig = (rightDatasetId: string, patch: Partial<DatasetJoinConfig>) => {
    setJoinPlan((current) =>
      current.map((config) =>
        config.right_dataset_id === rightDatasetId ? { ...config, ...patch } : config,
      ),
    );
  };

  const run = async () => {
    if (!selectedDatasetId) {
      setError("请先导入数据集。");
      return;
    }
    if (!question.trim()) {
      setError("请输入本次分析要回答的问题。");
      return;
    }
    if (!selectedDatasetGroup && selectedAdditionalDatasetIds.length && joinPlan.length !== selectedAdditionalDatasetIds.length) {
      setError("请先确认每个附加数据集的 join 配置。");
      return;
    }
    if (selectedDatasetGroup && !groupJoinPlan.length) {
      setError("当前数据包还没有通过自动校验的关系，请先在数据集页重新运行关系识别。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      let decision = plannerDecision;
      if (!decision || decision.semantic_plan.question !== question) {
        try {
          decision = await apiPost<PlannerDecision>("/analysis/plans", {
            dataset_id: selectedDatasetId,
            dataset_group_id: selectedDatasetGroup?.group_id ?? null,
            additional_dataset_ids: selectedDatasetGroup ? groupAdditionalDatasetIds : selectedAdditionalDatasetIds,
            question,
          });
          setPlannerDecision(decision);
        } catch {
          decision = null;
          setPlannerDecision(null);
        }
      }
      if (decision?.requires_confirmation && !confirmLowConfidence) {
        setError("语义计划置信度较低，请检查下方指标、维度和 Join 后确认继续。");
        return;
      }
      const multimodalInputs: MultimodalInput[] = [];
      if (multimodalContext.trim()) {
        multimodalInputs.push({
          kind: "note",
          title: "用户提供的多模态上下文",
          description: multimodalContext.trim(),
          source_ref: "analysis_form",
        });
      }
      if (multimodalFile) {
        if (multimodalFile.size > 5 * 1024 * 1024) {
          throw new Error("多模态附件不能超过 5MB。");
        }
        const isImage = multimodalFile.type.startsWith("image/");
        const isPdf =
          multimodalFile.type === "application/pdf" ||
          multimodalFile.name.toLowerCase().endsWith(".pdf");
        multimodalInputs.push({
          kind: isImage ? "screenshot" : "pdf_page",
          title: multimodalFile.name,
          description: multimodalContext.trim() || (isImage ? "用户上传的分析辅助图片。" : "用户上传的 PDF/文档页辅助材料。"),
          source_ref: "analysis_file_upload",
          media_type: multimodalFile.type || null,
          data_url: isImage || isPdf ? await fileToDataUrl(multimodalFile) : null,
        });
      }
      const job = await apiPost<AnalysisJob>(
        "/analysis/jobs",
        {
          dataset_id: selectedDatasetId,
          dataset_group_id: selectedDatasetGroup?.group_id ?? null,
          additional_dataset_ids: selectedDatasetGroup ? groupAdditionalDatasetIds : selectedAdditionalDatasetIds,
          join_plan: selectedDatasetGroup ? groupJoinPlan : selectedAdditionalDatasetIds.length ? joinPlan : [],
          relationship_plan: selectedDatasetGroup ? groupJoinPlan : [],
          question,
          multimodal_inputs: multimodalInputs,
          planner_decision_id: decision?.decision_id ?? null,
          confirmed_low_confidence: confirmLowConfidence,
          agent_mode: agentLoopEnabled ? "loop" : "legacy",
        },
      );
      updateCurrentJob(job);
      onSelectedJobIdChange(job.job_id);
      onLatestResultChange(null);
      await onJobsChanged();
      const finishedJob = await pollAnalysisJob(job.job_id, updateCurrentJob);
      await onJobsChanged();
      if (finishedJob.status !== "completed") {
        throw new Error(finishedJob.error || `分析任务${jobStatusLabel(finishedJob.status)}。`);
      }
      const payload = await apiGet<AnalysisResponse>(`/analysis/jobs/${finishedJob.job_id}/result`);
      onLatestResultChange(payload);
      await onReportsChanged();
    } catch (err) {
      setError(await analysisErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const cancelCurrentJob = async () => {
    if (!currentJob || !isActiveAnalysisJob(currentJob)) return;
    try {
      const job = await apiPost<AnalysisJob>(`/analysis/jobs/${currentJob.job_id}/cancel`, {});
      updateCurrentJob(job);
      await onJobsChanged();
    } catch (err) {
      setError(await analysisErrorMessage(err));
    }
  };

  const retryCurrentJob = async () => {
    if (!currentJob || isActiveAnalysisJob(currentJob)) return;
    setBusy(true);
    setError(null);
    try {
      const job = await apiPost<AnalysisJob>(`/analysis/jobs/${currentJob.job_id}/retry`, {});
      updateCurrentJob(job);
      onSelectedJobIdChange(job.job_id);
      onLatestResultChange(null);
      await onJobsChanged();
      const finishedJob = await pollAnalysisJob(job.job_id, updateCurrentJob);
      await onJobsChanged();
      if (finishedJob.status !== "completed") {
        throw new Error(finishedJob.error || `分析任务${jobStatusLabel(finishedJob.status)}。`);
      }
      const payload = await apiGet<AnalysisResponse>(`/analysis/jobs/${finishedJob.job_id}/result`);
      onLatestResultChange(payload);
      await onReportsChanged();
    } catch (err) {
      setError(await analysisErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const startNewSession = () => {
    onSelectedJobIdChange(null);
    onLatestResultChange(null);
    setCurrentJob(null);
    setQuestion("");
    setMultimodalContext("");
    setMultimodalFile(null);
    setError(null);
  };

  return (
    <section className={`analysis-session-layout ${historyCollapsed ? "is-history-collapsed" : ""}`}>
      <AnalysisSessionSidebar
        jobs={jobs}
        datasets={datasets}
        selectedJobId={selectedJobId}
        onSelect={onSelectedJobIdChange}
        onNew={startNewSession}
        collapsed={historyCollapsed}
        onToggleCollapsed={() => setHistoryCollapsed((current) => !current)}
      />
      <div className="analysis-session-workspace">
        <header className="analysis-session-header">
          <div>
            <span>{currentJob ? "分析会话" : "Agent Workspace"}</span>
            <h2>{currentJob?.question || "新建分析"}</h2>
            <p>
              {currentJob
                ? `${datasets.find((dataset) => dataset.dataset_id === currentJob.dataset_id)?.name ?? currentJob.dataset_id.slice(0, 8)} · ${formatTime(currentJob.created_at)}`
                : "选择数据并提交一个问题，每次运行都会创建独立的 Workflow 记录。"}
            </p>
          </div>
          {currentJob && <span className={`analysis-session-status is-${currentJob.status}`}>{jobStatusLabel(currentJob.status)}</span>}
        </header>

        {!currentJob && (
          <section className="analysis-compose-panel">
            <div className="analysis-form-grid">
              <div>
                <label className="label">数据包（可选）</label>
                <select value={selectedDatasetGroupId} onChange={(event) => selectDatasetGroup(event.target.value)} className="input">
                  <option value="">不使用数据包，按单个/手动多数据集分析</option>
                  {datasetGroups.map((group) => {
                    const relationshipCount = group.relationships.filter((relationship) => relationship.enabled !== false).length;
                    return (
                      <option key={group.group_id} value={group.group_id}>
                        {group.name} · {group.tables.length} 张表 · {relationshipCount ? `已自动配置 ${relationshipCount} 条关系` : "关系待识别"}
                      </option>
                    );
                  })}
                </select>
              </div>
              <div>
                <label className="label">{selectedDatasetGroup ? "主数据集（由关系计划确定）" : "主数据集"}</label>
                <select disabled={!!selectedDatasetGroup} value={selectedDatasetId} onChange={(event) => onActiveDatasetChange(event.target.value)} className="input">
                  {datasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.name}</option>)}
                </select>
              </div>
            </div>
            {selectedDatasetGroup && (
              <section className={`analysis-group-readiness ${relationshipBlocked ? "is-blocked" : "is-ready"}`}>
                <div className="analysis-group-readiness-heading">
                  <span>{relationshipBlocked ? <Network size={20} /> : <CircleCheckBig size={20} />}</span>
                  <div>
                    <h3>{relationshipBlocked ? "数据包关系尚未自动建立" : "数据包已准备完成"}</h3>
                    <p>
                      {relationshipBlocked
                        ? `${selectedDatasetGroup.tables.length} 张表尚无通过校验的 Join 关系，请返回数据集页重新运行自动识别。`
                        : `${selectedDatasetGroup.tables.length} 张表 · ${groupJoinPlan.length} 条自动保存关系，可开始分析。`}
                    </p>
                  </div>
                  <button type="button" onClick={onOpenDatasetRelationships}>
                    {relationshipBlocked ? "前往自动识别" : "查看关系"} <ArrowRight size={16} />
                  </button>
                </div>
                {relationshipBlocked ? (
                  <div className="analysis-relationship-steps">
                    <span><b>1</b>运行自动识别</span>
                    <span><b>2</b>规则与语义联合判断</span>
                    <span><b>3</b>校验后自动保存</span>
                  </div>
                ) : (
                  <div className="analysis-confirmed-relationships">
                    {selectedGroupRelationships.map((relationship) => (
                      <div key={relationshipKey(relationship)}>
                        {formatRelationship(relationship, selectedDatasetGroup.tables)} · 推荐置信度 {(Number(relationship.confidence ?? 0) * 100).toFixed(0)}%
                        {relationshipMatchRate(relationship) != null
                          ? ` · 样本匹配率 ${(relationshipMatchRate(relationship)! * 100).toFixed(0)}%`
                          : ""}
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
            {!selectedDatasetGroup && (
              <MultiDatasetJoinPanel
                primaryDatasetId={selectedDatasetId}
                datasets={datasets}
                additionalDatasetOptions={additionalDatasetOptions}
                selectedAdditionalDatasetIds={selectedAdditionalDatasetIds}
                joinSuggestions={joinSuggestions}
                joinPlan={joinPlan}
                joinReferences={joinReferences}
                busy={joinBusy}
                error={joinError}
                onToggleAdditionalDataset={toggleAdditionalDataset}
                onUpdateJoinConfig={updateJoinConfig}
              />
            )}
            <label className="label mt-5" htmlFor="analysis-question">分析问题</label>
            <textarea
              id="analysis-question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              className="input min-h-28 resize-none"
              placeholder="输入希望 DataMind 回答的业务问题"
            />
            <label className="label mt-4" htmlFor="analysis-context">补充上下文（可选）</label>
            <textarea
              id="analysis-context"
              value={multimodalContext}
              onChange={(event) => setMultimodalContext(event.target.value)}
              className="input min-h-20 resize-none"
              placeholder="补充截图、PDF 或业务背景中的关键信息"
            />
            <div className="analysis-attachment-row">
              <label>
                <FileText size={16} /> 选择图片或 PDF
                <input type="file" accept="image/*,.pdf" className="hidden" onChange={(event) => setMultimodalFile(event.target.files?.[0] ?? null)} />
              </label>
              <span>{multimodalFile ? `${multimodalFile.name} · ${(multimodalFile.size / 1024).toFixed(1)}KB` : "最大 5MB"}</span>
            </div>
            <div className="analysis-run-bar">
              <div className="analysis-run-summary">
                <span className="analysis-run-eyebrow"><Sparkles size={14} /> 当前执行路径</span>
                <b>{agentLoopEnabled ? "自主分析 Loop" : "兼容分析流程"}</b>
                <p>{agentLoopEnabled ? "规划器 → 结构理解 → 自主工具循环 → 证据整合 → 可视化 → 审查 → 报告" : "规划器 → 结构理解 → SQL → Python → 可视化 → 审查 → 报告"}</p>
                <small>{agentLoopEnabled ? "AI 会自主选择白名单工具，并在失败时修复、重试或安全降级。" : "使用固定 SQL/Python 路径，适合复现旧任务或排查兼容性问题。"}</small>
              </div>
              <div className="analysis-run-actions">
                <div className="analysis-mode-switch" role="radiogroup" aria-label="分析执行模式">
                  <button
                    type="button"
                    role="radio"
                    aria-checked={agentLoopEnabled}
                    className={agentLoopEnabled ? "active" : ""}
                    onClick={() => setAgentLoopEnabled(true)}
                  >
                    <Sparkles size={15} />
                    <span>自主分析<small>默认</small></span>
                  </button>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={!agentLoopEnabled}
                    className={!agentLoopEnabled ? "active compatibility" : ""}
                    onClick={() => setAgentLoopEnabled(false)}
                  >
                    <History size={15} />
                    <span>兼容模式</span>
                  </button>
                </div>
                <button disabled={busy || !datasets.length || relationshipBlocked} onClick={run} className="dashboard-primary-action">
                  {busy ? <Loader2 className="animate-spin" size={17} /> : <Play size={17} />}
                  {busy ? "正在创建任务" : relationshipBlocked ? "等待关系识别" : "开始分析"}
                </button>
              </div>
            </div>
            {plannerDecision && <SemanticPlanConfirmation decision={plannerDecision} confirmed={confirmLowConfidence} onConfirmed={setConfirmLowConfidence} />}
            {error && <Alert tone="error">{error}</Alert>}
          </section>
        )}

        {currentJob && (
          <div className="analysis-session-content">
            <section className="analysis-workflow-surface">
              <DynamicAgentPlan job={currentJob} />
              <RealtimeWorkflowPanel job={currentJob} />
              <AgentLoopPanel job={currentJob} />
              <AnalysisJobStatusPanel job={currentJob} onCancel={cancelCurrentJob} onRetry={retryCurrentJob} />
              {error && <Alert tone="error">{error}</Alert>}
            </section>
            <section className="analysis-result-surface">
              <div className="analysis-result-heading">
                <div>
                  <h3>分析输出</h3>
                  <p>SQL、Python 洞察、图表与最终报告</p>
                </div>
                {currentJob.report_id && <span>Report {currentJob.report_id.slice(0, 8)}</span>}
              </div>
              {isActiveAnalysisJob(currentJob) && <Alert>Workflow 正在执行，节点事件和运行日志会实时更新。</Alert>}
              {!isActiveAnalysisJob(currentJob) && currentJob.status !== "completed" && !latestResult && (
                <Alert tone="error">该任务未生成最终结果，可在上方查看失败节点和错误信息后重试。</Alert>
              )}
              {currentJob.status === "completed" && !latestResult && <LoadingLine />}
              {latestResult && <AnalysisResult result={latestResult} />}
            </section>
          </div>
        )}
      </div>
    </section>
  );
}

function SemanticPlanConfirmation({ decision, confirmed, onConfirmed }: { decision: PlannerDecision; confirmed: boolean; onConfirmed: (value: boolean) => void }) {
  const metrics = (decision.semantic_plan.metric_ids as string[] | undefined) ?? [];
  const dimensions = (decision.semantic_plan.dimension_ids as string[] | undefined) ?? [];
  return (
    <section className={`mt-4 rounded-xl border p-4 ${decision.confidence_level === "low" ? "border-amber-300 bg-amber-50" : "border-indigo-200 bg-indigo-50"}`}>
      <div className="flex items-center justify-between gap-3"><b>语义计划 · {decision.confidence_level.toUpperCase()}</b><span>{(decision.calibrated_confidence * 100).toFixed(0)}%</span></div>
      <p className="mt-2 text-sm font-semibold">指标：{metrics.join(", ") || "未解析"} · 维度：{dimensions.join(", ") || "未解析"}</p>
      <div className="mt-2 flex flex-wrap gap-2">{Object.entries(decision.confidence_breakdown).map(([key, value]) => value == null ? null : <span key={key} className="relationship-decision">{key} {(value * 100).toFixed(0)}%</span>)}</div>
      {!!decision.ambiguities.length && <p className="mt-2 text-sm font-bold text-amber-800">{decision.ambiguities.join("；")}</p>}
      {decision.requires_confirmation && <label className="mt-3 flex items-center gap-2 text-sm font-black"><input type="checkbox" checked={confirmed} onChange={(event) => onConfirmed(event.target.checked)} />我已确认该指标、维度和 Join 口径</label>}
    </section>
  );
}

function MultiDatasetJoinPanel({
  primaryDatasetId,
  datasets,
  additionalDatasetOptions,
  selectedAdditionalDatasetIds,
  joinSuggestions,
  joinPlan,
  joinReferences,
  busy,
  error,
  onToggleAdditionalDataset,
  onUpdateJoinConfig,
}: {
  primaryDatasetId: string;
  datasets: Dataset[];
  additionalDatasetOptions: Dataset[];
  selectedAdditionalDatasetIds: string[];
  joinSuggestions: JoinSuggestionResponse | null;
  joinPlan: DatasetJoinConfig[];
  joinReferences: Record<string, DatasetReference>;
  busy: boolean;
  error: string | null;
  onToggleAdditionalDataset: (datasetId: string) => void;
  onUpdateJoinConfig: (rightDatasetId: string, patch: Partial<DatasetJoinConfig>) => void;
}) {
  const [previewRowsByDataset, setPreviewRowsByDataset] = useState<Record<string, Record<string, unknown>[]>>({});
  const [previewBusyDatasetId, setPreviewBusyDatasetId] = useState<string | null>(null);
  const primaryReference = joinSuggestions?.primary_dataset ?? joinReferences[primaryDatasetId] ?? null;
  const references = new Map<string, DatasetReference>();
  Object.values(joinReferences).forEach((dataset) => references.set(dataset.dataset_id, dataset));
  if (joinSuggestions) {
    references.set(joinSuggestions.primary_dataset.dataset_id, joinSuggestions.primary_dataset);
    joinSuggestions.additional_datasets.forEach((dataset) => references.set(dataset.dataset_id, dataset));
  }
  const loadPreview = async (datasetId: string) => {
    setPreviewBusyDatasetId(datasetId);
    try {
      const payload = await apiGet<{ records: Record<string, unknown>[] }>(`/store/datasets/${datasetId}/preview?source=analysis&limit=5`);
      setPreviewRowsByDataset((current) => ({ ...current, [datasetId]: payload.records }));
    } finally {
      setPreviewBusyDatasetId(null);
    }
  };
  return (
    <section className="mb-4 rounded-xl border border-line bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="section-title mb-1">多文件分析 / Join</h3>
          <p className="text-sm text-slate-500">
            主数据集作为分析主体，可添加已上传数据集并确认关联字段。
          </p>
        </div>
        {busy && <span className="text-sm font-bold text-emerald-700">正在推荐 join key...</span>}
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {additionalDatasetOptions.map((dataset) => {
          const selected = selectedAdditionalDatasetIds.includes(dataset.dataset_id);
          return (
            <button
              key={dataset.dataset_id}
              type="button"
              className={selected ? "small-button is-selected-dataset" : "small-button"}
              aria-pressed={selected}
              title={selected ? `移除 ${dataset.name}` : `添加 ${dataset.name}`}
              onClick={() => onToggleAdditionalDataset(dataset.dataset_id)}
            >
              {selected ? "已添加 " : "添加 "}
              {dataset.name}
            </button>
          );
        })}
        {!additionalDatasetOptions.length && <span className="text-sm text-slate-500">暂无其他可用数据集。</span>}
      </div>
      {error && <Alert tone="error">{error}</Alert>}
      {primaryReference && (
        <div className="mt-3 rounded-xl border border-line bg-slate-50 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <b className="text-slate-950">主数据集：{primaryReference.name}</b>
              <p className="mt-1 text-xs font-bold text-slate-500">
                {primaryReference.row_count} 行 · {primaryReference.column_count} 列 · {primaryReference.columns.slice(0, 8).join(", ")}
              </p>
            </div>
            <button type="button" className="small-button" disabled={previewBusyDatasetId === primaryReference.dataset_id} onClick={() => void loadPreview(primaryReference.dataset_id)}>
              {previewBusyDatasetId === primaryReference.dataset_id ? "加载中" : "查看样本"}
            </button>
          </div>
          {previewRowsByDataset[primaryReference.dataset_id] && (
            <DataTable rows={previewRowsByDataset[primaryReference.dataset_id]} emptyText="暂无样本。" />
          )}
        </div>
      )}
      {!!joinSuggestions?.validation_issues.length && (
        <div className="mt-3 grid gap-2">
          {joinSuggestions.validation_issues.map((issue) => (
            <Alert key={issue.issue}>{issue.issue}</Alert>
          ))}
        </div>
      )}
      {!!selectedAdditionalDatasetIds.length && (
        <div className="mt-4 grid gap-3">
          {selectedAdditionalDatasetIds.map((rightDatasetId) => {
            const rightReference = references.get(rightDatasetId);
            const rightDataset = datasets.find((dataset) => dataset.dataset_id === rightDatasetId);
            const config =
              joinPlan.find((item) => item.right_dataset_id === rightDatasetId) ??
              ({
                left_dataset_id: primaryDatasetId,
                right_dataset_id: rightDatasetId,
                left_column: primaryReference?.columns[0] ?? "",
                right_column: rightReference?.columns[0] ?? "",
                join_type: "left",
              } satisfies DatasetJoinConfig);
            const candidates = (joinSuggestions?.suggestions ?? [])
              .filter((item) => item.right_dataset_id === rightDatasetId)
              .sort((a, b) => b.score - a.score);
            const leftColumns = primaryReference?.columns ?? [];
            const rightColumns = rightReference?.columns ?? [];
            return (
              <div key={rightDatasetId} className="rounded-xl border border-line bg-slate-50 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <b className="text-slate-950">{rightReference?.name ?? rightDataset?.name ?? rightDatasetId}</b>
                    <p className="mt-1 text-xs font-bold text-slate-500">
                      {rightReference
                        ? `${rightReference.row_count} 行 · ${rightReference.column_count} 列 · ${rightReference.columns.slice(0, 8).join(", ")}`
                        : "等待推荐"}
                    </p>
                  </div>
                  <button type="button" className="small-button" disabled={previewBusyDatasetId === rightDatasetId} onClick={() => void loadPreview(rightDatasetId)}>
                    {previewBusyDatasetId === rightDatasetId ? "加载中" : "查看样本"}
                  </button>
                </div>
                {previewRowsByDataset[rightDatasetId] && (
                  <DataTable rows={previewRowsByDataset[rightDatasetId]} emptyText="暂无样本。" />
                )}
                <div className="mt-3 grid gap-3 md:grid-cols-4">
                  <label className="text-xs font-black uppercase tracking-wide text-slate-500">
                    主表字段
                    <select
                      className="input mt-1 py-2"
                      value={config.left_column}
                      onChange={(event) => onUpdateJoinConfig(rightDatasetId, { left_column: event.target.value })}
                    >
                      {leftColumns.map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs font-black uppercase tracking-wide text-slate-500">
                    附表字段
                    <select
                      className="input mt-1 py-2"
                      value={config.right_column}
                      onChange={(event) => onUpdateJoinConfig(rightDatasetId, { right_column: event.target.value })}
                    >
                      {rightColumns.map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs font-black uppercase tracking-wide text-slate-500">
                    Join 类型
                    <select
                      className="input mt-1 py-2"
                      value={config.join_type}
                      onChange={(event) => onUpdateJoinConfig(rightDatasetId, { join_type: event.target.value as "left" | "inner" })}
                    >
                      <option value="left">left</option>
                      <option value="inner">inner</option>
                    </select>
                  </label>
                  <div className="rounded-lg border border-line bg-white px-3 py-2 text-xs leading-5 text-slate-600">
                    <b className="block text-slate-950">推荐</b>
                    {candidates[0]
                      ? `${candidates[0].left_column} -> ${candidates[0].right_column} · 推荐置信度 ${(candidates[0].score * 100).toFixed(0)}% · 样本匹配率 ${(candidates[0].estimated_match_rate * 100).toFixed(0)}%`
                      : "暂无推荐"}
                  </div>
                </div>
                {!!candidates.length && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {candidates.slice(0, 3).map((candidate) => (
                      <button
                        key={`${candidate.left_column}-${candidate.right_column}`}
                        type="button"
                        className="small-button"
                        onClick={() =>
                          onUpdateJoinConfig(rightDatasetId, {
                            left_column: candidate.left_column,
                            right_column: candidate.right_column,
                            join_type: "left",
                            left_value_mode: candidate.left_value_mode,
                            right_value_mode: candidate.right_value_mode,
                            left_delimiter: candidate.left_delimiter,
                            right_delimiter: candidate.right_delimiter,
                          })
                        }
                      >
                        {candidate.left_column} {"->"} {candidate.right_column} · 推荐置信度 {(candidate.score * 100).toFixed(0)}% · 样本匹配率 {(candidate.estimated_match_rate * 100).toFixed(0)}%
                        {candidate.left_value_mode === "delimited" || candidate.right_value_mode === "delimited" ? " · 列表键" : ""}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function AnalysisSessionSidebar({
  jobs,
  datasets,
  selectedJobId,
  onSelect,
  onNew,
  collapsed,
  onToggleCollapsed,
}: {
  jobs: AnalysisJob[];
  datasets: Dataset[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
  onNew: () => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const [query, setQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const historyListRef = useRef<HTMLDivElement | null>(null);
  const datasetById = useMemo(
    () => new Map(datasets.map((dataset) => [dataset.dataset_id, dataset])),
    [datasets],
  );
  const visibleJobs = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return jobs;
    return jobs.filter((job) => {
      const datasetName = datasetById.get(job.dataset_id)?.name ?? "";
      return `${job.question} ${datasetName} ${jobStatusLabel(job.status)}`.toLowerCase().includes(normalizedQuery);
    });
  }, [datasetById, jobs, query]);
  const expandAndFocus = (target: "search" | "history") => {
    if (collapsed) onToggleCollapsed();
    window.setTimeout(() => {
      if (target === "search") {
        searchInputRef.current?.focus();
        return;
      }
      const activeItem = historyListRef.current?.querySelector<HTMLButtonElement>(
        ".analysis-session-item.is-active, .analysis-session-item",
      );
      activeItem?.focus();
    }, 0);
  };
  return (
    <aside className={`analysis-session-sidebar ${collapsed ? "is-collapsed" : ""}`}>
      <div className="analysis-session-sidebar-header">
        <div className="analysis-session-sidebar-title">
          <span>History</span>
          <h2>分析记录</h2>
        </div>
        <div className="analysis-session-sidebar-actions">
          <button
            type="button"
            className="history-sidebar-toggle"
            onClick={onToggleCollapsed}
            aria-label={collapsed ? "展开分析记录" : "收起分析记录"}
            title={collapsed ? "展开分析记录" : "收起分析记录"}
          >
            {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
          <button className="analysis-header-new-button" type="button" onClick={onNew} aria-label="新建分析" title="新建分析"><Plus size={18} /></button>
        </div>
      </div>
      {collapsed && (
        <nav className="history-collapsed-tools" aria-label="分析记录快捷操作">
          <button type="button" className="history-collapsed-tool" data-tooltip="新建分析" aria-label="新建分析" onClick={onNew}>
            <SquarePen size={18} />
          </button>
          <button type="button" className="history-collapsed-tool" data-tooltip="搜索分析" aria-label="搜索分析" onClick={() => expandAndFocus("search")}>
            <Search size={18} />
          </button>
          <button type="button" className="history-collapsed-tool" data-tooltip="历史记录" aria-label="查看分析历史" onClick={() => expandAndFocus("history")}>
            <History size={18} />
          </button>
        </nav>
      )}
      <label className="analysis-session-search">
        <Search size={15} />
        <input ref={searchInputRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索分析记录" />
      </label>
      <button type="button" className={`analysis-session-new ${selectedJobId === null ? "is-active" : ""}`} onClick={onNew}>
        <span><Plus size={17} /></span>
        <span><b>新建分析</b><small>创建一个独立 Workflow</small></span>
      </button>
      <div ref={historyListRef} className="analysis-session-list">
        {visibleJobs.map((job) => {
          const dataset = datasetById.get(job.dataset_id);
          return (
            <button
              key={job.job_id}
              type="button"
              className={`analysis-session-item ${selectedJobId === job.job_id ? "is-active" : ""}`}
              onClick={() => onSelect(job.job_id)}
            >
              <span className={`analysis-session-item-icon is-${job.status}`}>
                {isActiveAnalysisJob(job) ? <Loader2 className="animate-spin" size={15} /> : <MessageSquareText size={15} />}
              </span>
              <span className="min-w-0 flex-1">
                <b>{job.question}</b>
                <small>{dataset?.name ?? job.dataset_id.slice(0, 8)}{job.additional_dataset_ids?.length ? ` +${job.additional_dataset_ids.length}` : ""}</small>
                <small>{formatTime(job.created_at)} · {jobStatusLabel(job.status)}</small>
                {isActiveAnalysisJob(job) && <i><span style={{ width: `${job.progress}%` }} /></i>}
              </span>
            </button>
          );
        })}
      </div>
      {!visibleJobs.length && <p className="analysis-session-empty">{jobs.length ? "没有匹配的记录" : "运行分析后会在这里生成记录"}</p>}
    </aside>
  );
}

function AnalysisResult({ result }: { result: AnalysisResponse }) {
  const structuredReport = structuredReportFromUnknown(result.structured_report);
  return (
    <div className="space-y-5">
      <MultimodalContextPanel inputs={result.multimodal_inputs ?? []} />
      <MultiDatasetContextPanel context={result.multi_dataset_context ?? null} />
      <PlannerMetadataPanel metadata={result.planner_metadata ?? null} />
      {!structuredReport && (
        <AnalysisReliabilityPanel
          contract={result.analysis_contract}
          verification={result.statistical_verification}
          lineage={result.analysis_lineage}
        />
      )}
      <WorkflowDebugger trace={result.workflow_trace ?? []} />
      {result.sql_result && (
        <section>
          <h3 className="section-title">生成的 SQL</h3>
          <p className="text-sm text-slate-500">执行来源: {result.sql_source ?? "rules"}</p>
          <pre className="code-block">{result.sql_result.sql}</pre>
          <DataTable rows={result.sql_result.rows} emptyText="SQL 没有返回行。" />
        </section>
      )}
      {result.python_result && (
        <section>
          <h3 className="section-title">Python Agent 洞察</h3>
          <p className="text-sm text-slate-500">执行来源: {result.python_source ?? "rules"}</p>
          <PythonAttemptsPanel
            attempts={result.python_attempts ?? []}
            executionError={result.python_execution_error ?? null}
            source={result.python_source ?? "rules"}
          />
          <ul className="mt-3 list-disc space-y-1 pl-5">
            {result.python_result.insights.map((insight) => (
              <li key={insight}>{insight}</li>
            ))}
          </ul>
          {result.python_generated_code && <pre className="code-block mt-4">{result.python_generated_code}</pre>}
          <TextAnalysisPanel results={result.python_result.text_analysis ?? []} />
          <ChartList charts={result.python_result.charts} />
        </section>
      )}
      {structuredReport ? (
        <section>
          <h3 className="section-title">结构化分析报告</h3>
          <StructuredReportPreview report={structuredReport} />
        </section>
      ) : result.report_markdown ? (
        <section>
          <h3 className="section-title">报告 Markdown</h3>
          <pre className="prose-block">{result.report_markdown}</pre>
        </section>
      ) : null}
    </div>
  );
}

function PythonAttemptsPanel({
  attempts,
  executionError,
  source,
}: {
  attempts: PythonCodeAttempt[];
  executionError: string | null;
  source: string;
}) {
  if (!attempts.length && !executionError) return null;
  const success = attempts.find((attempt) => attempt.status === "succeeded");
  const failedAfterRetries = !success && attempts.length >= 3 && executionError;
  return (
    <div className="mt-3 rounded-xl border border-line bg-slate-50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <b className="text-slate-950">代码执行自修复</b>
        <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-slate-600">
          {success ? `第 ${success.attempt} 次成功` : `${attempts.length || 1} 次尝试`}
        </span>
      </div>
      {success && (
        <p className="mt-2 text-sm text-emerald-700">
          LLM 生成代码在第 {success.attempt} 次执行成功。
        </p>
      )}
      {failedAfterRetries && (
        <Alert tone="error">
          LLM Python 代码连续 3 次执行失败，已使用规则 fallback 继续完成分析：{executionError}
        </Alert>
      )}
      {!success && source === "rules" && executionError && !failedAfterRetries && (
        <Alert tone="error">Python 回退：{executionError}</Alert>
      )}
      {!!attempts.length && (
        <details className="mt-3 rounded-lg border border-line bg-white px-3 py-2">
          <summary className="cursor-pointer font-black">查看每次代码与错误</summary>
          <div className="mt-3 grid gap-3">
            {attempts.map((attempt) => (
              <div key={attempt.attempt} className="rounded-lg border border-line bg-slate-50 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <b>第 {attempt.attempt} 次 · {attempt.status === "succeeded" ? "成功" : "失败"}</b>
                  <span className="text-xs font-bold text-slate-500">
                    {attempt.provider ?? "-"} / {attempt.model ?? "-"}
                  </span>
                </div>
                {attempt.error && <Alert tone="error">{attempt.error}</Alert>}
                {attempt.code && <pre className="code-block mt-3">{attempt.code}</pre>}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function MultiDatasetContextPanel({ context }: { context: MultiDatasetContext | null }) {
  if (!context) return null;
  const joins = arrayOfRecords(context.join_summary.joins);
  const sourceEntries = Object.entries(context.column_source_map ?? {});
  return (
    <section className="rounded-xl border border-cyan-100 bg-cyan-50/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="section-title mb-0">多文件分析上下文</h3>
        <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-cyan-800">
          {context.primary_dataset.name} + {context.additional_datasets.length}
        </span>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        <MetricPill label="主数据集" value={context.primary_dataset.name} />
        <MetricPill label="附加数据集" value={String(context.additional_datasets.length)} />
        <MetricPill
          label="已连接表"
          value={`${String(context.join_summary.joined_dataset_count ?? 1)}/${String(context.join_summary.dataset_count ?? context.additional_datasets.length + 1)}`}
        />
        <MetricPill label="Join 后行数" value={String(context.join_summary.joined_row_count ?? context.joined_profile?.row_count ?? "-")} />
        <MetricPill label="Join 后列数" value={String(context.join_summary.joined_column_count ?? context.joined_profile?.column_count ?? "-")} />
        <MetricPill label="总行数膨胀" value={`${Number(context.join_summary.row_expansion_ratio ?? 1).toFixed(2)}x`} />
      </div>
      {!!joins.length && (
        <div className="mt-3 overflow-auto rounded-lg border border-line bg-white">
          <table className="w-full min-w-[980px] border-collapse text-left text-sm">
            <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2">附加数据集</th>
                <th className="px-3 py-2">Join</th>
                <th className="px-3 py-2">状态</th>
                <th className="px-3 py-2">行数变化</th>
                <th className="px-3 py-2">膨胀 / 键</th>
                <th className="px-3 py-2">未匹配</th>
              </tr>
            </thead>
            <tbody>
              {joins.map((join, index) => (
                <tr key={index} className="border-t border-line">
                  <td className="px-3 py-2">{String(join.right_dataset_name ?? join.right_dataset_id ?? "-")}</td>
                  <td className="px-3 py-2">
                    {String(join.left_column ?? "-")} {"->"} {String(join.right_column ?? "-")} · {String(join.join_type ?? "-")}
                  </td>
                  <td className="px-3 py-2">
                    <span className={`relationship-decision ${join.status === "joined" ? "is-adopted" : ""}`}>
                      {join.status === "joined" ? "已执行" : "风险跳过"}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    {String(join.before_rows ?? "-")} {"->"} {String(join.after_rows ?? join.estimated_rows ?? "-")}
                  </td>
                  <td className="px-3 py-2">
                    {Number(join.row_expansion_ratio ?? join.estimated_expansion_ratio ?? 1).toFixed(2)}x · {join.right_key_unique ? "右键唯一" : "右键重复"}
                  </td>
                  <td className="px-3 py-2">{String(join.unmatched_rows ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!!sourceEntries.length && (
        <details className="mt-3 rounded-lg border border-line bg-white px-3 py-2">
          <summary className="cursor-pointer font-black">字段来源映射</summary>
          <div className="mt-2 flex flex-wrap gap-2">
            {sourceEntries.slice(0, 24).map(([column, source]) => (
              <span key={column} className="rounded-full bg-slate-100 px-2 py-1 text-xs font-bold text-slate-700">
                {column} · {source}
              </span>
            ))}
          </div>
        </details>
      )}
      {!!context.validation_issues.length && (
        <div className="mt-3 grid gap-2">
          {context.validation_issues.map((issue) => (
            <Alert key={issue.issue}>{issue.issue}</Alert>
          ))}
        </div>
      )}
    </section>
  );
}

function PlannerMetadataPanel({ metadata }: { metadata: PlannerMetadata | null }) {
  if (!metadata) return null;
  const rulesExecution = metadata.route_reason.includes("确定性规则");
  return (
    <section className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="section-title mb-0">Planner 元数据</h3>
        <div className="flex flex-wrap gap-2">
          {rulesExecution && (
            <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-black text-amber-800">
              规则执行
            </span>
          )}
          <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-emerald-800">
            置信度 {(metadata.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-700">{metadata.route_reason}</p>
      <div className="mt-3 grid gap-2 md:grid-cols-4">
        <MetricPill label="候选指标" value={metadata.candidate_metrics.slice(0, 3).join(", ") || "-"} />
        <MetricPill label="候选维度" value={metadata.candidate_dimensions.slice(0, 3).join(", ") || "-"} />
        <MetricPill label="时间字段" value={metadata.candidate_time_fields.slice(0, 3).join(", ") || "-"} />
        <MetricPill label="文本字段" value={metadata.candidate_text_fields.slice(0, 3).join(", ") || "-"} />
      </div>
      {!!metadata.clarifying_questions.length && (
        <div className="mt-3 grid gap-2">
          {metadata.clarifying_questions.map((question) => (
            <Alert key={question}>{question}</Alert>
          ))}
        </div>
      )}
    </section>
  );
}

function WorkflowDebugger({ trace }: { trace: WorkflowTraceNode[] }) {
  if (!trace.length) return null;
  return (
    <section className="rounded-xl border border-line bg-white p-4">
      <h3 className="section-title">Workflow 调试视图</h3>
      <div className="grid gap-3">
        {trace.map((node) => (
          <details key={`${node.node}-${node.status}`} className="rounded-xl border border-line bg-slate-50 px-4 py-3">
            <summary className="cursor-pointer font-black">
              {jobStageLabel(node.node)} · {node.status}
              {node.fallback ? ` · fallback=${node.fallback}` : ""}
            </summary>
            <div className="mt-3 grid gap-2 text-sm leading-6 text-slate-700 md:grid-cols-2">
              <div className="rounded-lg bg-white p-3">
                <b className="block text-slate-950">输入摘要</b>
                {node.input_summary || "-"}
              </div>
              <div className="rounded-lg bg-white p-3">
                <b className="block text-slate-950">输出摘要</b>
                {node.output_summary || "-"}
              </div>
              <div className="rounded-lg bg-white p-3">
                <b className="block text-slate-950">模型</b>
                {node.provider || "-"} / {node.model || "-"}
              </div>
              <div className="rounded-lg bg-white p-3">
                <b className="block text-slate-950">错误</b>
                {node.error || "-"}
              </div>
            </div>
          </details>
        ))}
      </div>
    </section>
  );
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("文件读取失败。"));
    reader.readAsDataURL(file);
  });
}
