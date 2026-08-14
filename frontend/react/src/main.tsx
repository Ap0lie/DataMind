import React, { useCallback, useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { FloatingTaskProgress, Sidebar, Topbar } from "./components/AppShell";
import { AssistantPage } from "./features/assistant/AssistantPage";
import type { AssistantRun } from "./features/assistant/types";
import { AnalysisPage } from "./features/analysis/AnalysisPage";
import { LoginPage } from "./features/auth/LoginPage";
import { DashboardPage } from "./features/dashboard/DashboardPage";
import { DatasetsPage } from "./features/datasets/DatasetsPage";
import { ReportsPage } from "./features/reports/ReportsPage";
import {
  AUTH_EXPIRED_EVENT,
  abortPendingApiRequests,
  apiGet,
  apiPost,
  loadAuthUser,
  logoutSession,
  saveAuthUser,
  type AuthUser,
  type CleaningJob,
} from "./api-client";
import { isActiveAnalysisJob, jobStageLabel } from "./workflow-ui";
import type {
  ActiveTask,
  AnalysisJob,
  AnalysisResponse,
  ApiState,
  Dataset,
  DatasetGroup,
  DatasetWorkspaceView,
  Page,
  Report,
} from "./domain-types";
import {
  assistantStageLabel,
  cleaningStageLabel,
  errorMessage,
  formatTime,
} from "./formatters";

const LATEST_ANALYSIS_STORAGE_KEY = "datamind.latestAnalysisResult.v1";
function App() {
  const [authUser, setAuthUser] = useState<AuthUser | null>(() => loadAuthUser());
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("首页");
  const [datasets, setDatasets] = useState<ApiState<Dataset[]>>({
    data: null,
    loading: true,
    error: null,
  });
  const [datasetGroups, setDatasetGroups] = useState<ApiState<DatasetGroup[]>>({
    data: null,
    loading: true,
    error: null,
  });
  const [reportSummaries, setReportSummaries] = useState<ApiState<Report[]>>({
    data: null,
    loading: true,
    error: null,
  });
  const [reports, setReports] = useState<ApiState<Report[]>>({
    data: null,
    loading: true,
    error: null,
  });
  const [analysisJobs, setAnalysisJobs] = useState<ApiState<AnalysisJob[]>>({
    data: null,
    loading: true,
    error: null,
  });
  const [cleaningJobs, setCleaningJobs] = useState<ApiState<CleaningJob[]>>({
    data: null,
    loading: true,
    error: null,
  });
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [datasetWorkspaceView, setDatasetWorkspaceView] = useState<DatasetWorkspaceView>("assets");
  const [selectedAnalysisJobId, setSelectedAnalysisJobId] = useState<string | null>(null);
  const [latestAnalysisResult, setLatestAnalysisResult] = useState<AnalysisResponse | null>(() =>
    loadLatestAnalysisResult(),
  );
  const [assistantActiveRuns, setAssistantActiveRuns] = useState(0);
  const [assistantActiveRun, setAssistantActiveRun] = useState<{
    run: AssistantRun;
    progress: number;
  } | null>(null);
  const handleAssistantActiveRunChange = useCallback(
    (run: AssistantRun | null, progress: number) =>
      setAssistantActiveRun(run ? { run, progress } : null),
    [],
  );

  const refresh = async () => {
    if (!authUser) return;
    await Promise.all([
      loadDatasets(),
      loadDatasetGroups(),
      loadReportSummaries(),
      loadReports(),
      loadAnalysisJobs(),
      loadCleaningJobs(),
    ]);
  };

  const loadDatasets = async () => {
    setDatasets((state) => ({ ...state, loading: true, error: null }));
    try {
      const payload = await apiGet<{ datasets: Dataset[] }>("/store/datasets");
      setDatasets({ data: payload.datasets, loading: false, error: null });
      setActiveDatasetId((current) =>
        current && payload.datasets.some((dataset) => dataset.dataset_id === current)
          ? current
          : payload.datasets[0]?.dataset_id ?? null,
      );
    } catch (error) {
      setDatasets((state) => ({ data: state.data, loading: false, error: errorMessage(error) }));
    }
  };

  const loadDatasetGroups = async () => {
    setDatasetGroups((state) => ({ ...state, loading: true, error: null }));
    try {
      const payload = await apiGet<{ groups: DatasetGroup[] }>("/store/dataset-groups");
      setDatasetGroups({ data: payload.groups, loading: false, error: null });
    } catch (error) {
      setDatasetGroups((state) => ({ data: state.data, loading: false, error: errorMessage(error) }));
    }
  };

  const loadReportSummaries = async () => {
    setReportSummaries((state) => ({ ...state, loading: true, error: null }));
    try {
      const payload = await apiGet<{ reports: Report[] }>("/store/reports?include_content=false");
      setReportSummaries({ data: payload.reports, loading: false, error: null });
    } catch (error) {
      setReportSummaries((state) => ({ data: state.data, loading: false, error: errorMessage(error) }));
    }
  };

  const loadReports = async () => {
    setReports((state) => ({ ...state, loading: true, error: null }));
    try {
      const payload = await apiGet<{ reports: Report[] }>("/store/reports");
      setReports({ data: payload.reports, loading: false, error: null });
    } catch (error) {
      setReports((state) => ({ data: state.data, loading: false, error: errorMessage(error) }));
    }
  };

  const loadAnalysisJobs = async () => {
    setAnalysisJobs((state) => ({ ...state, loading: true, error: null }));
    try {
      const payload = await apiGet<{ jobs: AnalysisJob[] }>("/analysis/jobs?limit=100");
      setAnalysisJobs({ data: payload.jobs, loading: false, error: null });
    } catch (error) {
      const message = errorMessage(error);
      if (message.includes("404")) {
        setAnalysisJobs({ data: [], loading: false, error: null });
        return;
      }
      setAnalysisJobs((state) => ({ data: state.data, loading: false, error: message }));
    }
  };

  const loadCleaningJobs = async () => {
    try {
      const payload = await apiGet<{ jobs: CleaningJob[] }>("/store/cleaning-jobs?limit=100");
      setCleaningJobs({ data: payload.jobs, loading: false, error: null });
    } catch (error) {
      setCleaningJobs((state) => ({
        data: state.data,
        loading: false,
        error: errorMessage(error),
      }));
    }
  };

  const updateAnalysisJob = (job: AnalysisJob) => {
    setAnalysisJobs((state) => {
      const current = state.data ?? [];
      const exists = current.some((item) => item.job_id === job.job_id);
      const data = exists
        ? current.map((item) => (item.job_id === job.job_id ? job : item))
        : [job, ...current];
      return { data, loading: false, error: state.error };
    });
  };

  const updateCleaningJob = (job: CleaningJob) => {
    setCleaningJobs((state) => {
      const current = state.data ?? [];
      const exists = current.some((item) => item.job_id === job.job_id);
      return {
        data: exists
          ? current.map((item) => (item.job_id === job.job_id ? job : item))
          : [job, ...current],
        loading: false,
        error: state.error,
      };
    });
  };

  const login = async (username: string, password: string) => {
    const user = await apiPost<AuthUser & { created: boolean }>("/auth/login", { username, password });
    const nextUser = {
      user_id: user.user_id,
      display_name: user.display_name,
      csrf_token: user.csrf_token ?? null,
      expires_at: user.expires_at ?? null,
    };
    saveAuthUser(nextUser);
    setSessionNotice(null);
    setAuthUser(nextUser);
    setActiveDatasetId(null);
    setLatestAnalysisResult(loadLatestAnalysisResult());
  };

  const clearLocalSession = () => {
    saveAuthUser(null);
    saveLatestAnalysisResult(null);
    setAuthUser(null);
    setLatestAnalysisResult(null);
    setActiveDatasetId(null);
    setDatasetWorkspaceView("assets");
    setSelectedAnalysisJobId(null);
    setDatasets({ data: null, loading: false, error: null });
    setDatasetGroups({ data: null, loading: false, error: null });
    setReports({ data: null, loading: false, error: null });
    setReportSummaries({ data: null, loading: false, error: null });
    setAnalysisJobs({ data: null, loading: false, error: null });
    setCleaningJobs({ data: null, loading: false, error: null });
    setAssistantActiveRun(null);
    setPage("首页");
  };

  const logout = async () => {
    abortPendingApiRequests();
    flushSync(() => setAuthUser(null));
    try {
      const result = await logoutSession();
      if (result === "logged_out") setSessionNotice(null);
    } catch (error) {
      console.warn("Server logout failed; clearing the local session.", error);
    }
    clearLocalSession();
  };

  useEffect(() => {
    if (authUser) void refresh();
  }, [authUser?.user_id]);

  useEffect(() => {
    saveLatestAnalysisResult(latestAnalysisResult);
  }, [latestAnalysisResult]);

  useEffect(() => {
    if (!authUser) return;
    const hasActiveCleaning = (cleaningJobs.data ?? []).some((job) =>
      ["queued", "running", "cancel_requested"].includes(job.status),
    );
    if (!hasActiveCleaning && assistantActiveRuns === 0) return;
    const timer = window.setInterval(() => void loadCleaningJobs(), 2500);
    return () => window.clearInterval(timer);
  }, [authUser?.user_id, assistantActiveRuns, cleaningJobs.data]);

  useEffect(() => {
    const handleExpiredSession = () => {
      setSessionNotice("登录状态已过期，请重新登录。");
      clearLocalSession();
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, handleExpiredSession);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, handleExpiredSession);
  }, []);

  const datasetList = datasets.data ?? [];
  const datasetGroupList = datasetGroups.data ?? [];
  const reportList = reports.data ?? [];
  const summaryList = reportSummaries.data ?? [];
  const jobList = analysisJobs.data ?? [];
  const cleaningJobList = cleaningJobs.data ?? [];
  const activeJobList = jobList
    .filter((job) => isActiveAnalysisJob(job))
    .sort((left, right) => String(right.updated_at ?? right.created_at ?? "").localeCompare(String(left.updated_at ?? left.created_at ?? "")));
  const activeCleaningJobList = cleaningJobList.filter((job) =>
    ["queued", "running", "cancel_requested"].includes(job.status),
  );
  const activeTasks: ActiveTask[] = [
    ...activeJobList.map((job) => ({
      id: job.job_id,
      kind: "analysis" as const,
      page: "分析任务" as const,
      title: job.question,
      stage: jobStageLabel(job.current_stage),
      progress: job.progress,
      updatedAt: String(job.updated_at ?? job.created_at ?? ""),
    })),
    ...activeCleaningJobList.map((job) => ({
      id: job.job_id,
      kind: "cleaning" as const,
      page: "数据集" as const,
      title: `${datasetList.find((dataset) => dataset.dataset_id === job.dataset_id)?.name ?? "数据集"} 正在清洗`,
      stage: cleaningStageLabel(job.current_stage),
      progress: job.progress,
      updatedAt: String(job.updated_at ?? job.created_at ?? ""),
    })),
  ];
  if (assistantActiveRuns > 0) {
    activeTasks.push({
      id: assistantActiveRun?.run.run_id ?? "assistant-active",
      kind: "assistant",
      page: "Kimi",
      title: "Kimi 正在处理当前任务",
      stage: assistantActiveRun
        ? assistantStageLabel(assistantActiveRun.run.current_stage)
        : "Kimi 正在后台运行",
      progress: assistantActiveRun?.progress ?? 8,
      updatedAt: assistantActiveRun?.run.updated_at ?? "",
    });
  }
  activeTasks.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  const floatingTask = activeTasks.find((task) => task.page !== page) ?? null;

  if (!authUser) {
    return <LoginPage notice={sessionNotice} onLogin={login} />;
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#f8fbff_0,#eef3f8_42%,#e9f0f7_100%)] text-ink">
      <Sidebar page={page} user={authUser} assistantActiveRuns={assistantActiveRuns} onPageChange={setPage} onLogout={logout} />
      <main className={`min-h-screen min-w-0 overflow-x-clip px-4 py-5 pb-24 md:ml-[236px] md:px-10 md:py-8 md:pb-8 ${page === "Kimi" ? "assistant-main h-screen overflow-y-hidden" : ""}`}>
        {page !== "Kimi" && <Topbar user={authUser} />}
        <div className={`mx-auto ${page === "Kimi" ? "assistant-main-content" : page === "分析任务" || page === "报告" ? "max-w-[1480px]" : "max-w-[1180px]"}`}>
          {page === "首页" && (
            <DashboardPage
              datasets={datasetList}
              reports={summaryList}
              jobs={jobList}
              loading={datasets.loading || reportSummaries.loading || analysisJobs.loading}
              error={datasets.error ?? reportSummaries.error ?? analysisJobs.error}
              onNavigate={setPage}
              onOpenAnalysis={(jobId) => {
                setSelectedAnalysisJobId(jobId);
                setPage("分析任务");
              }}
              onStartAnalysis={() => {
                setSelectedAnalysisJobId(null);
                setLatestAnalysisResult(null);
                setPage("分析任务");
              }}
              onRetry={refresh}
            />
          )}
          <div className={page === "数据集" ? "" : "hidden"} aria-hidden={page !== "数据集"}>
            <DatasetsPage
              datasets={datasetList}
              datasetGroups={datasetGroupList}
              activeDatasetId={activeDatasetId}
              onActiveDatasetChange={setActiveDatasetId}
              onRefresh={refresh}
              loading={datasets.loading}
              error={datasets.error}
              workspaceView={datasetWorkspaceView}
              onWorkspaceViewChange={setDatasetWorkspaceView}
              onCleaningJobUpdate={updateCleaningJob}
            />
          </div>
          <div className={page === "分析任务" ? "" : "hidden"} aria-hidden={page !== "分析任务"}>
            <AnalysisPage
              datasets={datasetList}
              datasetGroups={datasetGroupList}
              activeDatasetId={activeDatasetId}
              onActiveDatasetChange={setActiveDatasetId}
              onReportsChanged={refresh}
              jobs={jobList}
              onJobsChanged={loadAnalysisJobs}
              latestResult={latestAnalysisResult}
              onLatestResultChange={setLatestAnalysisResult}
              selectedJobId={selectedAnalysisJobId}
              onSelectedJobIdChange={setSelectedAnalysisJobId}
              onJobUpdate={updateAnalysisJob}
              onOpenDatasetRelationships={() => {
                setDatasetWorkspaceView("relationships");
                setPage("数据集");
              }}
            />
          </div>
          {page === "报告" && (
            <ReportsPage
              datasets={datasetList}
              datasetGroups={datasetGroupList}
              reports={reportList}
              loading={reports.loading}
              error={reports.error}
              onRefresh={refresh}
            />
          )}
          <div className={page === "Kimi" ? "" : "hidden"} aria-hidden={page !== "Kimi"}>
            <AssistantPage
              datasets={datasetList.map((dataset) => ({ id: dataset.dataset_id, name: dataset.name }))}
              datasetGroups={datasetGroupList.map((group) => ({ id: group.group_id, name: group.name }))}
              reports={summaryList.map((report) => ({
                id: report.id,
                name: report.title,
                description: `v${report.version ?? 1} · ${formatTime(report.created_at)}`,
              }))}
              onActiveRunsChange={setAssistantActiveRuns}
              onActiveRunChange={handleAssistantActiveRunChange}
              onAssetsChanged={refresh}
              onOpenDataset={(datasetId) => {
                setActiveDatasetId(datasetId);
                setDatasetWorkspaceView("detail");
                setPage("数据集");
              }}
              onOpenAnalysis={(jobId) => {
                setSelectedAnalysisJobId(jobId);
                setPage("分析任务");
              }}
              onOpenReport={() => setPage("报告")}
            />
          </div>
        </div>
      </main>
      {floatingTask && (
        <FloatingTaskProgress
          task={floatingTask}
          activeCount={activeTasks.length}
          onOpen={() => {
            if (floatingTask.kind === "analysis") {
              setSelectedAnalysisJobId(floatingTask.id);
            }
            setPage(floatingTask.page);
          }}
        />
      )}
    </div>
  );
}

function loadLatestAnalysisResult(): AnalysisResponse | null {
  try {
    const raw = window.localStorage.getItem(LATEST_ANALYSIS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<AnalysisResponse>;
    if (!parsed || typeof parsed !== "object") return null;
    if (typeof parsed.dataset_id !== "string" || typeof parsed.question !== "string") return null;
    return {
      dataset_id: parsed.dataset_id,
      report_id: parsed.report_id ?? null,
      question: parsed.question,
      multimodal_inputs: Array.isArray(parsed.multimodal_inputs) ? parsed.multimodal_inputs : [],
      planner_metadata: parsed.planner_metadata ?? null,
      analysis_contract: parsed.analysis_contract ?? null,
      statistical_verification: parsed.statistical_verification ?? null,
      analysis_lineage: parsed.analysis_lineage ?? null,
      multi_dataset_context: parsed.multi_dataset_context ?? null,
      sql_result: parsed.sql_result ?? null,
      python_result: parsed.python_result ?? null,
      structured_report: parsed.structured_report ?? null,
      html_report: null,
      report_markdown: typeof parsed.report_markdown === "string" ? parsed.report_markdown : "",
      sql_source: parsed.sql_source ?? null,
      python_source: parsed.python_source ?? null,
      python_generated_code: parsed.python_generated_code ?? null,
      python_execution_error: parsed.python_execution_error ?? null,
      python_attempts: Array.isArray(parsed.python_attempts) ? parsed.python_attempts : [],
      workflow_trace: Array.isArray(parsed.workflow_trace) ? parsed.workflow_trace : [],
    };
  } catch (error) {
    console.warn("Failed to restore latest analysis result.", error);
    return null;
  }
}

function saveLatestAnalysisResult(result: AnalysisResponse | null) {
  try {
    if (!result) {
      window.localStorage.removeItem(LATEST_ANALYSIS_STORAGE_KEY);
      return;
    }
    const stored: AnalysisResponse = { ...result, html_report: null };
    window.localStorage.setItem(LATEST_ANALYSIS_STORAGE_KEY, JSON.stringify(stored));
  } catch (error) {
    console.warn("Failed to persist latest analysis result.", error);
  }
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
