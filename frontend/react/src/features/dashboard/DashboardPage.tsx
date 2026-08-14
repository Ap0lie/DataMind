import {
  Activity,
  ArrowRight,
  BarChart3,
  CircleCheckBig,
  Database,
  FileText,
  Loader2,
  Play,
  RefreshCw,
  TriangleAlert,
  Upload,
} from "lucide-react";
import { Alert, Metric } from "../../components/primitives";
import type { AnalysisJob, Dataset, Page, Report } from "../../domain-types";
import {
  dashboardSyncErrorMessage,
  formatTime,
  latestReportCaption,
  translateStatus,
} from "../../formatters";
import {
  deriveAnalysisComponents,
  isActiveAnalysisJob,
  jobStageLabel,
  jobStatusLabel,
} from "../../workflow-ui";

export function DashboardPage({
  datasets,
  reports,
  jobs,
  loading,
  error,
  onNavigate,
  onOpenAnalysis,
  onStartAnalysis,
  onRetry,
}: {
  datasets: Dataset[];
  reports: Report[];
  jobs: AnalysisJob[];
  loading: boolean;
  error: string | null;
  onNavigate: (page: Page) => void;
  onOpenAnalysis: (jobId: string) => void;
  onStartAnalysis: () => void;
  onRetry: () => Promise<void>;
}) {
  const datasetById = new Map(datasets.map((dataset) => [dataset.dataset_id, dataset]));
  const reportById = new Map(reports.map((report) => [report.id, report]));
  const recentJobs = jobs.slice(0, 8);
  const activeJobs = jobs.filter((job) => isActiveAnalysisJob(job));
  const completedJobs = jobs.filter((job) => job.status === "completed");
  const completedComponents = completedJobs.map((job) =>
    deriveAnalysisComponents(
      job,
      job.report_id ? reportById.get(job.report_id)?.metadata : undefined,
    ),
  );
  const sqlCount = completedComponents.filter((components) => components.sql).length;
  const pythonCount = completedComponents.filter((components) => components.python).length;
  const hybridCount = completedComponents.filter(
    (components) => components.sql && components.python,
  ).length;
  const failedJobs = jobs.filter((job) => ["failed", "interrupted"].includes(job.status));
  const cleanedDatasets = datasets.filter((dataset) => dataset.status === "cleaned");
  const readiness = datasets.length
    ? Math.round((cleanedDatasets.length / datasets.length) * 100)
    : 0;
  const reportRouteTotal = Math.max(sqlCount + pythonCount, 1);

  return (
    <section className="dashboard-page">
      <div className="dashboard-header">
        <div>
          <div className="dashboard-status-line">
            <Activity size={15} />
            {activeJobs.length ? `${activeJobs.length} 个分析任务正在运行` : "工作区已就绪"}
          </div>
          <h2 className="page-title mb-2">工作区</h2>
          <p>集中查看数据准备状态、分析任务和最近产出的报告。</p>
        </div>
        <div className="dashboard-actions">
          <button type="button" className="dashboard-secondary-action" onClick={() => onNavigate("数据集")}>
            <Upload size={17} /> 导入数据
          </button>
          <button type="button" className="dashboard-primary-action" onClick={onStartAnalysis}>
            <Play size={17} /> 开始分析 <ArrowRight size={16} />
          </button>
        </div>
      </div>
      {error && (
        <div className="dashboard-sync-alert" role="alert" aria-live="assertive">
          <span className="dashboard-sync-alert-icon"><TriangleAlert size={18} /></span>
          <span className="dashboard-sync-alert-copy">
            <strong>部分数据暂时未同步</strong>
            <small>{dashboardSyncErrorMessage(error)}</small>
          </span>
          <button type="button" disabled={loading} onClick={() => void onRetry()}>
            <RefreshCw className={loading ? "animate-spin" : ""} size={16} />
            {loading ? "同步中" : "重新同步"}
          </button>
        </div>
      )}
      <div className="dashboard-metrics">
        <Metric icon={<Database size={18} />} label="数据资产" value={datasets.length} caption={`${cleanedDatasets.length} 个已清洗`} />
        <Metric icon={<BarChart3 size={18} />} label="已完成分析" value={completedJobs.length} caption={`SQL ${sqlCount} · Python ${pythonCount}`} />
        <Metric icon={<FileText size={18} />} label="分析报告" value={reports.length} caption={latestReportCaption(reports)} />
        <Metric icon={<Play size={18} />} label="任务队列" value={activeJobs.length} caption={failedJobs.length ? `${failedJobs.length} 个任务需要关注` : "没有失败任务"} />
      </div>
      <div className="dashboard-content-grid">
        <section className="dashboard-panel dashboard-activity-panel">
          <div className="dashboard-panel-heading">
            <div>
              <h3>最近活动</h3>
              <p>运行中的任务与最近完成的分析</p>
            </div>
            <button type="button" onClick={() => onNavigate("分析任务")}>全部任务 <ArrowRight size={15} /></button>
          </div>
          {loading && (
            <div className="dashboard-panel-loading">
              <Loader2 className="animate-spin" size={17} /> 正在加载最近活动...
            </div>
          )}
          {!loading && !recentJobs.length && (
            <Alert>还没有分析记录。上传数据集并运行分析后会显示在这里。</Alert>
          )}
          <div className="dashboard-activity-list">
            {recentJobs.map((job) => {
              const dataset = datasetById.get(job.dataset_id);
              return (
                <button key={job.job_id} type="button" className="dashboard-activity-row" onClick={() => onOpenAnalysis(job.job_id)}>
                  <span className={`dashboard-activity-icon is-${job.status}`}>
                    {isActiveAnalysisJob(job)
                      ? <Loader2 className="animate-spin" size={17} />
                      : job.status === "completed"
                        ? <CircleCheckBig size={17} />
                        : "!"}
                  </span>
                  <span className="min-w-0 flex-1 text-left">
                    <b>{job.question}</b>
                    <small>{dataset?.name ?? job.dataset_id.slice(0, 8)} · {jobStageLabel(job.current_stage)} · {formatTime(job.updated_at)}</small>
                    {job.error && <small className="text-rose-700">{job.error}</small>}
                  </span>
                  <span className={`dashboard-job-status is-${job.status}`}>{jobStatusLabel(job.status)}</span>
                  <ArrowRight className="shrink-0 text-slate-300" size={16} />
                </button>
              );
            })}
          </div>
        </section>
        <div className="dashboard-side-stack">
          <section className="dashboard-panel">
            <div className="dashboard-panel-heading">
              <div>
                <h3>数据准备</h3>
                <p>{cleanedDatasets.length} / {datasets.length} 个数据集可用于分析</p>
              </div>
              <strong>{readiness}%</strong>
            </div>
            <div className="dashboard-progress" aria-label={`数据准备进度 ${readiness}%`}>
              <span style={{ width: `${readiness}%` }} />
            </div>
            {!datasets.length && <Alert>还没有导入的数据集。</Alert>}
            {!!datasets.length && (
              <div className="dashboard-dataset-list">
                {datasets.slice(0, 5).map((dataset) => (
                  <button key={dataset.dataset_id} type="button" onClick={() => onNavigate("数据集")}>
                    <span className="dashboard-dataset-icon"><Database size={16} /></span>
                    <span className="min-w-0 flex-1">
                      <b>{dataset.name}</b>
                      <small>{dataset.source_type.toUpperCase()} · {formatTime(dataset.created_at)}</small>
                    </span>
                    <span className={`dataset-status ${dataset.status === "cleaned" ? "is-cleaned" : ""}`}>{translateStatus(dataset.status)}</span>
                  </button>
                ))}
              </div>
            )}
          </section>
          <section className="dashboard-panel">
            <div className="dashboard-panel-heading">
              <div>
                <h3>分析构成</h3>
                <p>{hybridCount} 次同时使用 SQL 与 Python</p>
              </div>
            </div>
            <div className="dashboard-route-list">
              <div><span>SQL</span><div><i style={{ width: `${(sqlCount / reportRouteTotal) * 100}%` }} /></div><b>{sqlCount}</b></div>
              <div><span>Python</span><div><i style={{ width: `${(pythonCount / reportRouteTotal) * 100}%` }} /></div><b>{pythonCount}</b></div>
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}
