import { useEffect, useMemo, useRef, useState } from "react";
import {
  FilePlus2,
  FileText,
  History,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
} from "lucide-react";
import { apiGet, apiPatch, apiPost } from "../../api-client";
import { Alert, LoadingLine } from "../../components/primitives";
import type {
  AnalysisJob,
  Dataset,
  DatasetGroup,
  DatasetJoinConfig,
  Report,
  ReportVersionSummary,
} from "../../domain-types";
import { errorMessage, formatTime } from "../../formatters";
import { isActiveAnalysisJob, jobStatusLabel } from "../../workflow-ui";
import { AnalysisJobStatusPanel } from "../analysis/WorkflowStatus";
import { analysisErrorMessage, pollAnalysisJob } from "../analysis/job-client";
import {
  DownloadButton,
  MultimodalContextPanel,
  ReportVersionCompare,
  StructuredReportPreview,
  htmlReportForDownload,
  multimodalInputsFromMetadata,
  structuredReportFromMetadata,
} from "./ReportContent";
import {
  markdownReportForTemplate,
  reportChartPrompt,
  reportForTemplate,
  reportTemplateLabel,
  reportTemplatePrompt,
  type ReportTemplateMode,
} from "./report-templates";

export function ReportsPage({
  datasets,
  datasetGroups,
  reports,
  loading,
  error,
  onRefresh,
}: {
  datasets: Dataset[];
  datasetGroups: DatasetGroup[];
  reports: Report[];
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<void>;
}) {
  const [dataScopeId, setDataScopeId] = useState("");
  const [question, setQuestion] = useState("分析数据中的主要变化和异常");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<AnalysisJob | null>(null);
  const [query, setQuery] = useState("");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reportVersions, setReportVersions] = useState<ReportVersionSummary[]>([]);
  const [renameTitle, setRenameTitle] = useState("");
  const [compareReport, setCompareReport] = useState<Report | null>(null);
  const [detailBusy, setDetailBusy] = useState(false);
  const [reportTemplate, setReportTemplate] = useState<ReportTemplateMode>("standard");
  const [historyCollapsed, setHistoryCollapsed] = useState(
    () => window.localStorage.getItem("datamind.reportHistoryCollapsed.v1") === "true",
  );
  const reportSearchInputRef = useRef<HTMLInputElement | null>(null);
  const reportHistoryListRef = useRef<HTMLDivElement | null>(null);
  const reportCreatePanelRef = useRef<HTMLDetailsElement | null>(null);
  const reportDatasetSelectRef = useRef<HTMLSelectElement | null>(null);
  const defaultDataScopeId = datasetGroups[0]
    ? `group:${datasetGroups[0].group_id}`
    : datasets[0]
      ? `dataset:${datasets[0].dataset_id}`
      : "";
  const selectedDataScopeId = dataScopeId || defaultDataScopeId;
  const selectedDatasetGroup = selectedDataScopeId.startsWith("group:")
    ? datasetGroups.find((group) => group.group_id === selectedDataScopeId.slice("group:".length)) ?? null
    : null;
  const selectedStandaloneDatasetId = selectedDataScopeId.startsWith("dataset:")
    ? selectedDataScopeId.slice("dataset:".length)
    : "";
  const selectedGroupRelationships = (selectedDatasetGroup?.relationships ?? []).filter(
    (relationship) => relationship.enabled !== false,
  );
  const relationshipRightDatasetIds = new Set(
    selectedGroupRelationships.map((relationship) => relationship.right_dataset_id),
  );
  const groupPrimaryDatasetId =
    selectedGroupRelationships.find(
      (relationship) => !relationshipRightDatasetIds.has(relationship.left_dataset_id),
    )?.left_dataset_id
    ?? selectedGroupRelationships[0]?.left_dataset_id
    ?? selectedDatasetGroup?.tables[0]?.dataset.dataset_id
    ?? null;
  const selectedDatasetId = groupPrimaryDatasetId || selectedStandaloneDatasetId || datasets[0]?.dataset_id || "";
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
        .filter((candidateId) => candidateId !== selectedDatasetId),
    ),
  );
  const filteredReports = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return reports;
    return reports.filter((report) => {
      const text = `${report.title} ${report.markdown} ${JSON.stringify(report.metadata)}`.toLowerCase();
      return text.includes(normalizedQuery);
    });
  }, [reports, query]);

  useEffect(() => {
    if (!selectedReport && reports.length) {
      void openReport(reports[0].id);
    }
  }, [reports.length]);

  useEffect(() => {
    window.localStorage.setItem("datamind.reportHistoryCollapsed.v1", String(historyCollapsed));
  }, [historyCollapsed]);

  const expandReportHistoryAndFocus = (target: "search" | "history") => {
    if (historyCollapsed) setHistoryCollapsed(false);
    window.setTimeout(() => {
      if (target === "search") {
        reportSearchInputRef.current?.focus();
        return;
      }
      const activeItem = reportHistoryListRef.current?.querySelector<HTMLButtonElement>(
        ".report-history-item.is-active, .report-history-item",
      );
      activeItem?.focus();
    }, 0);
  };

  const openReportCreator = () => {
    if (reportCreatePanelRef.current) reportCreatePanelRef.current.open = true;
    reportCreatePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    window.setTimeout(() => reportDatasetSelectRef.current?.focus(), 0);
  };

  const openReport = async (reportId: string) => {
    setDetailBusy(true);
    try {
      const summary = reports.find((report) => report.id === reportId);
      const reportPromise = apiGet<Report>(`/store/reports/${reportId}`);
      const versionsPromise = summary
        ? apiGet<{ versions: ReportVersionSummary[] }>(`/store/datasets/${summary.dataset_id}/report-versions`)
        : null;
      const report = await reportPromise;
      setSelectedReport(report);
      setCompareReport(null);
      setRenameTitle(report.title);
      const versions = versionsPromise
        ? await versionsPromise
        : await apiGet<{ versions: ReportVersionSummary[] }>(`/store/datasets/${report.dataset_id}/report-versions`);
      setReportVersions(versions.versions);
    } catch (err) {
      setMessage(errorMessage(err));
    } finally {
      setDetailBusy(false);
    }
  };

  const openCompareReport = async (reportId: string) => {
    setDetailBusy(true);
    try {
      setCompareReport(await apiGet<Report>(`/store/reports/${reportId}`));
    } catch (err) {
      setMessage(errorMessage(err));
    } finally {
      setDetailBusy(false);
    }
  };

  const renameReport = async () => {
    if (!selectedReport) return;
    setDetailBusy(true);
    try {
      const report = await apiPatch<Report>(`/store/reports/${selectedReport.id}`, { title: renameTitle });
      setSelectedReport(report);
      setRenameTitle(report.title);
      await onRefresh();
    } catch (err) {
      setMessage(errorMessage(err));
    } finally {
      setDetailBusy(false);
    }
  };

  const generate = async () => {
    if (!selectedDatasetId) {
      setMessage("请先上传数据集。");
      return;
    }
    if (selectedDatasetGroup && !groupJoinPlan.length) {
      setMessage("当前数据包还没有通过自动校验的关系，请先在数据集页重新运行关系识别。");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const job = await apiPost<AnalysisJob>("/analysis/jobs", {
        dataset_id: selectedDatasetId,
        dataset_group_id: selectedDatasetGroup?.group_id ?? null,
        additional_dataset_ids: selectedDatasetGroup ? groupAdditionalDatasetIds : [],
        join_plan: selectedDatasetGroup ? groupJoinPlan : [],
        relationship_plan: selectedDatasetGroup ? groupJoinPlan : [],
        question,
        multimodal_inputs: [],
        prompt_overrides: {
          report: reportTemplatePrompt(reportTemplate),
          visualization: reportChartPrompt(reportTemplate),
        },
      });
      setCurrentJob(job);
      const finishedJob = await pollAnalysisJob(job.job_id, setCurrentJob);
      if (finishedJob.status !== "completed") {
        throw new Error(finishedJob.error || `报告任务${jobStatusLabel(finishedJob.status)}。`);
      }
      await onRefresh();
      setMessage("报告已生成并保存到数据库。");
    } catch (err) {
      setMessage(await analysisErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const cancelCurrentJob = async () => {
    if (!currentJob || !isActiveAnalysisJob(currentJob)) return;
    try {
      const job = await apiPost<AnalysisJob>(`/analysis/jobs/${currentJob.job_id}/cancel`, {});
      setCurrentJob(job);
    } catch (err) {
      setMessage(await analysisErrorMessage(err));
    }
  };

  const retryCurrentJob = async () => {
    if (!currentJob || isActiveAnalysisJob(currentJob)) return;
    setBusy(true);
    setMessage(null);
    try {
      const job = await apiPost<AnalysisJob>(`/analysis/jobs/${currentJob.job_id}/retry`, {});
      setCurrentJob(job);
      const finishedJob = await pollAnalysisJob(job.job_id, setCurrentJob);
      if (finishedJob.status !== "completed") {
        throw new Error(finishedJob.error || `报告任务${jobStatusLabel(finishedJob.status)}。`);
      }
      await onRefresh();
      setMessage("报告已生成并保存到数据库。");
    } catch (err) {
      setMessage(await analysisErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="reports-page">
      <div className="reports-page-heading">
        <div>
          <h2 className="page-title">报告</h2>
          <p>集中查看、管理与导出已生成的分析结果</p>
        </div>
        <span>{reports.length} 份报告</span>
      </div>

      <details ref={reportCreatePanelRef} className="report-create-panel">
        <summary className="report-create-summary">
          <span className="report-create-icon"><Plus size={18} /></span>
          <span className="min-w-0">
            <b>生成新报告</b>
            <small>选择数据包或数据集并输入分析问题</small>
          </span>
          <span className="report-create-toggle">展开</span>
        </summary>
        <div className="report-create-body">
          <div className="report-create-fields">
            <label>
              <span>数据范围</span>
              <select ref={reportDatasetSelectRef} value={selectedDataScopeId} onChange={(event) => setDataScopeId(event.target.value)} className="input">
                {!!datasetGroups.length && (
                  <optgroup label="数据包（支持跨表分析）">
                    {datasetGroups.map((group) => (
                      <option key={group.group_id} value={`group:${group.group_id}`}>
                        {group.name} · {group.tables.length} 张表
                      </option>
                    ))}
                  </optgroup>
                )}
                {!!datasets.length && (
                  <optgroup label="单个数据集">
                    {datasets.map((dataset) => (
                      <option key={dataset.dataset_id} value={`dataset:${dataset.dataset_id}`}>
                        {dataset.name}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </label>
            <label>
              <span>分析问题</span>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} className="input min-h-12 resize-y" />
            </label>
            <label>
              <span>报告模板</span>
              <select value={reportTemplate} onChange={(event) => setReportTemplate(event.target.value as ReportTemplateMode)} className="input">
                <option value="brief">简报 · 核心结论优先</option>
                <option value="standard">标准 · 结论与证据平衡</option>
                <option value="detailed">详细 · 包含完整轨迹</option>
              </select>
            </label>
            <button onClick={generate} disabled={busy} className="report-create-submit">
              {busy ? <Loader2 className="animate-spin" size={16} /> : <FileText size={16} />}
              生成报告
            </button>
          </div>
          {currentJob && (
            <AnalysisJobStatusPanel
              job={currentJob}
              onCancel={cancelCurrentJob}
              onRetry={retryCurrentJob}
            />
          )}
          {message && <Alert>{message}</Alert>}
        </div>
      </details>

      <div className={`report-workspace ${historyCollapsed ? "is-history-collapsed" : ""}`}>
        <aside className={`report-history-panel ${historyCollapsed ? "is-collapsed" : ""}`}>
          <div className="report-history-heading">
            <div className="report-history-title">
              <span>REPORTS</span>
              <h3>历史报告</h3>
            </div>
            <div className="report-history-controls">
              <b>{filteredReports.length}</b>
              <button
                type="button"
                className="history-sidebar-toggle"
                onClick={() => setHistoryCollapsed((current) => !current)}
                aria-label={historyCollapsed ? "展开历史报告" : "收起历史报告"}
                title={historyCollapsed ? "展开历史报告" : "收起历史报告"}
              >
                {historyCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
              </button>
            </div>
          </div>
          {historyCollapsed && (
            <nav className="history-collapsed-tools" aria-label="历史报告快捷操作">
              <button type="button" className="history-collapsed-tool" data-tooltip="生成新报告" aria-label="生成新报告" onClick={openReportCreator}>
                <FilePlus2 size={18} />
              </button>
              <button type="button" className="history-collapsed-tool" data-tooltip="搜索报告" aria-label="搜索报告" onClick={() => expandReportHistoryAndFocus("search")}>
                <Search size={18} />
              </button>
              <button type="button" className="history-collapsed-tool" data-tooltip="历史报告" aria-label="查看历史报告" onClick={() => expandReportHistoryAndFocus("history")}>
                <History size={18} />
              </button>
            </nav>
          )}
          <label className="report-history-search">
            <Search size={16} />
            <input
              ref={reportSearchInputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索标题、问题或正文"
            />
          </label>
          {loading && <LoadingLine />}
          {error && <Alert tone="error">{error}</Alert>}
          {!loading && !reports.length && <Alert>还没有生成报告。</Alert>}
          <div ref={reportHistoryListRef} className="report-history-list">
          {filteredReports.map((report) => (
            <button
              key={report.id}
              type="button"
              className={`report-history-item ${selectedReport?.id === report.id ? "is-active" : ""}`}
              onClick={() => void openReport(report.id)}
            >
              <span className="report-history-item-icon"><FileText size={16} /></span>
              <span className="min-w-0 flex-1">
                <b>{report.title}</b>
                <small>v{report.version ?? 1} · {formatTime(report.created_at)}</small>
                <small>{String(report.metadata?.question ?? "")}</small>
              </span>
            </button>
          ))}
          {!filteredReports.length && !loading && <Alert>没有匹配的报告。</Alert>}
          </div>
        </aside>
        <div className="report-detail-workspace report-print-area">
          {detailBusy && <LoadingLine />}
          {selectedReport ? (() => {
          const report = selectedReport;
          const structuredReport = structuredReportFromMetadata(report.metadata);
          const displayReport = structuredReport ? reportForTemplate(structuredReport, reportTemplate) : null;
          const multimodalInputs = multimodalInputsFromMetadata(report.metadata);
          const htmlContent = displayReport
            ? htmlReportForDownload(report.title, displayReport)
            : typeof report.metadata?.html_report === "string"
              ? String(report.metadata.html_report)
              : "";
          const markdownContent = displayReport
            ? markdownReportForTemplate(report.title, displayReport)
            : report.markdown;
          return (
          <article className="report-card" key={report.id}>
            <div className="report-detail-header report-toolbar">
              <div className="min-w-0">
                <h3 className="text-2xl font-black text-slate-950">{report.title}</h3>
                <p className="text-sm font-bold text-slate-500">
                  v{report.version ?? 1} · 创建 {formatTime(report.created_at)} · 更新 {formatTime(report.updated_at)}
                </p>
              </div>
              <button type="button" className="small-button" onClick={() => window.print()}>
                导出 PDF
              </button>
            </div>
            <div className="report-actions report-toolbar">
              <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-1" aria-label="报告模板">
                {(["brief", "standard", "detailed"] as ReportTemplateMode[]).map((mode) => (
                  <button key={mode} type="button" className={`rounded-md px-3 py-2 text-sm font-black ${reportTemplate === mode ? "bg-white text-emerald-800 shadow-sm" : "text-slate-500"}`} onClick={() => setReportTemplate(mode)}>
                    {reportTemplateLabel(mode)}
                  </button>
                ))}
              </div>
              <input value={renameTitle} onChange={(event) => setRenameTitle(event.target.value)} className="report-rename-input" aria-label="报告标题" />
              <button type="button" className="small-button" disabled={detailBusy} onClick={() => void renameReport()}>
                重命名
              </button>
              {htmlContent && (
                <DownloadButton
                  label="导出 HTML 报告"
                  fileName={`${report.title}_${report.dataset_id.slice(0, 8)}.html`}
                  content={htmlContent}
                  mime="text/html"
                />
              )}
              <DownloadButton
                label="导出 Markdown"
                fileName={`${report.title}_${report.dataset_id.slice(0, 8)}.md`}
                content={markdownContent}
                mime="text/markdown"
              />
            </div>
            {!!reportVersions.length && (
              <details className="report-version-panel report-toolbar">
                <summary className="cursor-pointer font-black">报告版本历史</summary>
                <div className="mt-3 grid gap-2">
                  {reportVersions.map((version) => (
                    <div key={version.report_id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-white px-3 py-2 text-sm font-bold text-slate-700">
                      <button type="button" className="text-left" onClick={() => void openReport(version.report_id)}>
                        v{version.version} · {version.title} · {formatTime(version.created_at)}
                      </button>
                      {version.report_id !== report.id && (
                        <button type="button" className="small-button h-8 px-3" onClick={() => void openCompareReport(version.report_id)}>
                          对比
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </details>
            )}
            {compareReport && <ReportVersionCompare left={compareReport} right={report} />}
            <MultimodalContextPanel inputs={multimodalInputs} compact />
            {displayReport ? (
              <StructuredReportPreview report={displayReport} />
            ) : (
              <pre className="prose-block">{report.markdown}</pre>
            )}
          </article>
          );
        })() : <Alert>从左侧选择一份报告查看详情。</Alert>}
        </div>
      </div>
    </section>
  );
}
