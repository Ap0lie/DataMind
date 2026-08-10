import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowRight,
  BarChart3,
  CircleCheckBig,
  Database,
  Download,
  Eye,
  FilePlus2,
  FileText,
  History,
  Home,
  Loader2,
  LogOut,
  MessageSquareText,
  Network,
  PackageOpen,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  SquarePen,
  Table2,
  TriangleAlert,
  Trash2,
  Upload,
} from "lucide-react";
import "./styles.css";
import { AgentLoopPanel } from "./agent-loop-ui";
import { CleaningLoopPanel } from "./cleaning-loop-ui";
import { AssistantPage } from "./features/assistant/AssistantPage";
import type { AssistantRun } from "./features/assistant/types";
import {
  AnalysisReliabilityPanel,
  type AnalysisContract,
  type AnalysisLineage,
  type StatisticalVerification,
} from "./features/analysis/AnalysisReliabilityPanel";
import { DriftMonitorPanel } from "./features/data-reliability/DriftMonitorPanel";
import { SemanticModelWorkbench } from "./features/semantic/SemanticModelWorkbench";
import {
  CleaningRuleEditor as DatasetCleaningRuleEditor,
  CleaningVersionsPanel as DatasetCleaningVersionsPanel,
  ColumnMetadataEditor as DatasetColumnMetadataEditor,
  type CleaningRule,
  type CleaningRulePreviewResponse,
  type CleaningRunDetail,
} from "./features/datasets/CleaningWorkspace";
import { exportChart } from "./features/reports/chart-export";
import {
  markdownReportForTemplate,
  reportChartPrompt,
  reportForTemplate,
  reportTemplateLabel,
  reportTemplatePrompt,
  type ReportTemplateMode,
} from "./features/reports/report-templates";
import {
  API_BASE_URL,
  AUTH_EXPIRED_EVENT,
  apiDelete,
  apiGet,
  apiPatch,
  apiPost,
  apiPostForm,
  loadAuthUser,
  logoutSession,
  runDatasetCleaning,
  saveAuthUser,
  type AuthUser,
  type CleaningJob,
} from "./api-client";
import {
  AGENT_PLAN_STEPS,
  agentStatusClass,
  buildWorkflowLogEntries,
  combinedWorkflowStatus,
  deriveAnalysisComponents,
  deriveAgentWorkflowViews,
  isActiveAnalysisJob,
  jobStageLabel,
  jobStatusLabel,
  translateWorkflowEventMessage,
  type AgentWorkflowStepKey,
  type WorkflowNodeStatus,
} from "./workflow-ui";

const LATEST_ANALYSIS_STORAGE_KEY = "datamind.latestAnalysisResult.v1";
const CHART_SERIES_COLORS = [
  "#5B7DB1",
  "#E07A5F",
  "#2A9D8F",
  "#E9B949",
  "#7A6FA8",
  "#4BA3C7",
  "#D96C88",
  "#78A06A",
  "#D58B4E",
  "#6B8E9E",
] as const;
const CHART_LINE_COLOR = "#4F6FAE";

type Page = "首页" | "数据集" | "分析任务" | "报告" | "Kimi";
type DatasetWorkspaceView = "assets" | "relationships" | "detail";

type Dataset = {
  dataset_id: string;
  user_id?: string;
  name: string;
  source_type: string;
  status: string;
  source_metadata: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

type DatasetColumnProfile = {
  name: string;
  dtype: string;
  missing_count: number;
  distinct_count: number;
  is_numeric: boolean;
  min_value?: number | null;
  max_value?: number | null;
  mean?: number | null;
};

type DatasetProfile = {
  dataset_id: string;
  row_count: number;
  column_count: number;
  missing_value_count: number;
  missing_value_ratio: number;
  duplicate_row_count: number;
  numeric_columns: string[];
  categorical_columns: string[];
  columns: DatasetColumnProfile[];
  sample_records: Record<string, unknown>[];
};

type DatasetJoinConfig = {
  left_dataset_id: string;
  right_dataset_id: string;
  left_column: string;
  right_column: string;
  join_type: "left" | "inner";
  left_value_mode?: "scalar" | "delimited";
  right_value_mode?: "scalar" | "delimited";
  left_delimiter?: string | null;
  right_delimiter?: string | null;
};

type DatasetRelationshipPlan = DatasetJoinConfig & {
  relationship_id?: string | null;
  enabled?: boolean;
  confidence?: number;
  source?: string;
  reason?: string;
  relationship_type?: "one_to_one" | "one_to_many" | "many_to_one" | "many_to_many" | "unknown";
  risk_note?: string;
  baseline_match_rate?: number | null;
  last_match_rate?: number | null;
  match_rate_drift?: number;
  freshness_status?: "fresh" | "warning" | "stale";
  stale_reason?: string;
  last_validated_at?: string | null;
  drift_event_id?: string | null;
};

type DatasetReference = {
  dataset_id: string;
  name: string;
  status: string;
  row_count: number;
  column_count: number;
  columns: string[];
};

type JoinSuggestionCandidate = DatasetJoinConfig & {
  score: number;
  reason: string;
  left_type: string;
  right_type: string;
  left_role: string;
  right_role: string;
  estimated_match_rate: number;
};

type DatasetRelationshipCandidate = DatasetRelationshipPlan & {
  confidence: number;
  source: "rules" | "llm" | "validated_llm";
  estimated_match_rate: number;
  left_type?: string;
  right_type?: string;
  left_role?: string;
  right_role?: string;
};

type DatasetGroupTable = {
  dataset: Dataset;
  row_count: number;
  column_count: number;
  columns: string[];
  entity_type: "fact" | "dimension" | "bridge" | "lookup" | "wide" | "unknown";
  sample_records: Record<string, unknown>[];
};

type DatasetGroup = {
  group_id: string;
  user_id?: string;
  name: string;
  description: string;
  tables: DatasetGroupTable[];
  relationships: DatasetRelationshipPlan[];
  metadata: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

type DatasetRelationshipSuggestionResponse = {
  group: DatasetGroup;
  candidates: DatasetRelationshipCandidate[];
  llm_used: boolean;
  compact_context: Record<string, unknown>;
  validation_issues: string[];
};

type DatasetRelationshipAutoConfigureResponse = DatasetRelationshipSuggestionResponse & {
  saved_relationships: DatasetRelationshipPlan[];
  primary_dataset_id: string | null;
  unresolved_dataset_ids: string[];
};

type PlannerDecision = {
  decision_id: string; semantic_source: string; semantic_model_id?: string | null;
  semantic_model_version?: number | null; semantic_plan: Record<string, unknown>;
  confidence_breakdown: Record<string, number | null>; raw_confidence: number;
  calibrated_confidence: number; confidence_level: "low" | "medium" | "high";
  requires_confirmation: boolean; ambiguities: string[]; evidence: string[];
};

type JoinSuggestionResponse = {
  primary_dataset: DatasetReference;
  additional_datasets: DatasetReference[];
  suggestions: JoinSuggestionCandidate[];
  validation_issues: { severity: string; finding_ref: string; issue: string; suggestion?: string }[];
};

type MultiDatasetContext = {
  primary_dataset: DatasetReference;
  additional_datasets: DatasetReference[];
  join_plan: DatasetJoinConfig[];
  join_summary: Record<string, unknown>;
  joined_profile?: DatasetProfile | null;
  column_source_map: Record<string, string>;
  validation_issues: { severity: string; finding_ref: string; issue: string; suggestion?: string }[];
};

type DatasetDetail = {
  profile: DatasetProfile | null;
  rawRecords: Record<string, unknown>[];
  cleanedRecords: Record<string, unknown>[];
  analysisRecords: Record<string, unknown>[];
  cleaningRuns: CleaningRunDetail[];
  columnMetadata: DatasetColumnMetadata[];
};

type Report = {
  id: string;
  dataset_id: string;
  title: string;
  markdown: string;
  metadata: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
  version?: number;
};

type DatasetColumnMetadata = {
  column_name: string;
  inferred_type: string;
  override_type?: string | null;
  description: string;
  role: "dimension" | "metric" | "id" | "text" | "date" | "ignore";
  created_at?: string | null;
  updated_at?: string | null;
};

type ReportVersionSummary = {
  report_id: string;
  dataset_id: string;
  title: string;
  question?: string | null;
  version: number;
  created_at?: string | null;
  updated_at?: string | null;
};

type ExcelSheetPreview = {
  sheet_name: string;
  row_count: number;
  column_count: number;
  score: number;
  selected: boolean;
  preview_records: Record<string, unknown>[];
};

type UploadQueueItem = {
  id: string;
  file: File;
  sheetPreviews: ExcelSheetPreview[];
  selectedSheetName: string;
  status: "ready" | "previewing" | "uploading" | "cleaning" | "done" | "error";
  message?: string;
  inserted?: number;
  datasetId?: string;
};

type DatasetImportPipelineState = {
  phase: "idle" | "processing" | "creating_group" | "relationships" | "complete" | "attention" | "error";
  message: string;
  tableCount: number;
  relationshipCount: number;
  unresolvedCount: number;
  llmUsed: boolean;
};

type Chart = {
  title: string;
  chart_type: string;
  spec: Record<string, unknown>;
  data: Record<string, unknown>[];
  explanation?: string;
  related_finding_ids?: string[];
};

type TextAnalysisResult = {
  task: string;
  text_column: string;
  group_column?: string | null;
  summary: Record<string, unknown>;
  insights: string[];
  charts: Chart[];
};

type PythonCodeAttempt = {
  attempt: number;
  status: "failed" | "succeeded";
  code?: string | null;
  error?: string | null;
  provider?: string | null;
  model?: string | null;
};

type StructuredReport = {
  executive_summary: string;
  analysis_context?: string;
  key_findings?: {
    title: string;
    content: string;
    data_source?: string;
    evidence?: string;
    confidence?: string;
    business_impact?: string;
    recommended_action?: string;
  }[];
  charts?: Chart[];
  chart_explanations?: string[];
  sql_results?: Record<string, unknown>[];
  python_results?: Record<string, unknown>;
  data_gaps?: string[];
  validation_issues?: { severity: string; finding_ref: string; issue: string; suggestion?: string }[];
  recommended_next_steps?: string[];
  analysis_trace?: {
    round_number: number;
    hypothesis?: { statement?: string };
    plan?: { route?: string; metric_column?: string | null; category_column?: string | null; time_column?: string | null };
    reflection?: { insight_text?: string; decision?: string };
    execution_result?: Record<string, unknown>;
    validation_status?: string;
  }[];
  analysis_contract?: AnalysisContract | null;
  statistical_verification?: StatisticalVerification | null;
  analysis_lineage?: AnalysisLineage | null;
};

type AnalysisResponse = {
  dataset_id: string;
  dataset_group_id?: string | null;
  report_id?: string | null;
  question: string;
  multimodal_inputs?: MultimodalInput[];
  planner_metadata?: PlannerMetadata | null;
  analysis_contract?: AnalysisContract | null;
  statistical_verification?: StatisticalVerification | null;
  analysis_lineage?: AnalysisLineage | null;
  multi_dataset_context?: MultiDatasetContext | null;
  sql_result?: { sql: string; rows: Record<string, unknown>[]; explanation: string } | null;
  python_result?: {
    statistics: Record<string, unknown>;
    insights: string[];
    charts: Chart[];
    text_analysis?: TextAnalysisResult[];
  } | null;
  structured_report?: Record<string, unknown> | null;
  html_report?: string | null;
  report_markdown: string;
  sql_source?: string | null;
  python_source?: string | null;
  python_generated_code?: string | null;
  python_execution_error?: string | null;
  python_attempts?: PythonCodeAttempt[];
  workflow_trace?: WorkflowTraceNode[];
  agent_mode?: "legacy" | "loop";
  loop_summary?: Record<string, unknown>;
  loop_terminal_reason?: string | null;
  report_strategy?: string | null;
  report_revision_count?: number;
  report_terminal_reason?: string | null;
};

type PlannerMetadata = {
  confidence: number;
  route_reason: string;
  candidate_metrics: string[];
  candidate_dimensions: string[];
  candidate_time_fields: string[];
  candidate_text_fields: string[];
  clarifying_questions: string[];
  multi_dataset_summary?: Record<string, unknown>;
};

type WorkflowTraceNode = {
  node: string;
  status: string;
  provider?: string | null;
  model?: string | null;
  input_summary: string;
  output_summary: string;
  fallback?: string | null;
  error?: string | null;
};

type AnalysisJobEvent = {
  sequence?: number;
  node?: string;
  stage: string;
  progress: number;
  message: string;
  status: string;
  created_at: string;
  attempt?: number;
  duration_ms?: number | null;
  provider?: string | null;
  model?: string | null;
  token_usage?: Record<string, number>;
  error_code?: string | null;
  event_type?: string | null;
  iteration?: number | null;
  tool_name?: string | null;
  repair_of_sequence?: number | null;
  payload?: Record<string, unknown>;
};

type AnalysisJob = {
  job_id: string;
  dataset_id: string;
  dataset_group_id?: string | null;
  additional_dataset_ids?: string[];
  join_plan?: DatasetJoinConfig[];
  relationship_plan?: DatasetJoinConfig[];
  question: string;
  status: string;
  progress: number;
  current_stage: string;
  events: AnalysisJobEvent[];
  error?: string | null;
  report_id?: string | null;
  retry_of?: string | null;
  cancel_requested?: boolean;
  attempt?: number;
  resumable?: boolean;
  last_event_sequence?: number;
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  agent_mode?: "legacy" | "loop";
  loop_summary?: Record<string, unknown>;
  loop_terminal_reason?: string | null;
};

type MultimodalInput = {
  kind: "image" | "chart" | "pdf_page" | "screenshot" | "note";
  title: string;
  description: string;
  source_ref?: string | null;
  media_type?: string | null;
  data_url?: string | null;
  processing_status?: string | null;
  text_excerpt?: string | null;
};

type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

type ActiveTask = {
  id: string;
  kind: "analysis" | "cleaning" | "assistant";
  page: Page;
  title: string;
  stage: string;
  progress: number;
  updatedAt: string;
};

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
            <Dashboard
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

function LoginPage({
  notice,
  onLogin,
}: {
  notice?: string | null;
  onLogin: (username: string, password: string) => Promise<void>;
}) {
  const [username, setUsername] = useState("default");
  const [password, setPassword] = useState("");
  const [focus, setFocus] = useState<"idle" | "username" | "password">("idle");
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!username.trim()) {
      setError("请输入用户名或邮箱。");
      return;
    }
    if (!password) {
      setError("请输入密码。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onLogin(username, password);
    } catch (err) {
      setError(loginErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main
      className="login-shell"
      onMouseMove={(event) =>
        setPointer({
          x: event.clientX / Math.max(window.innerWidth, 1) - 0.5,
          y: event.clientY / Math.max(window.innerHeight, 1) - 0.5,
        })
      }
    >
      <section className="login-panel">
        <div className="character-stage" aria-hidden="true">
          <LoginCharacter tone="purple" focus={focus} pointer={pointer} />
          <LoginCharacter tone="dark" focus={focus} pointer={pointer} delay />
          <LoginCharacter tone="orange" focus={focus} pointer={pointer} half />
          <LoginCharacter tone="yellow" focus={focus} pointer={pointer} small />
        </div>
        <form className="login-form" onSubmit={submit}>
          <div>
            <p className="label mb-3">DATA ANALYSIS AGENT</p>
            <h1 className="text-4xl font-black tracking-normal text-slate-950">Welcome to DataMind</h1>
            <p className="mt-3 max-w-md text-sm font-semibold leading-6 text-slate-500">
              登录后你的数据集、清洗结果、分析报告会和其他用户隔离保存。
            </p>
          </div>
          <label className="label mt-8" htmlFor="datamind-username">用户名 / 邮箱</label>
          <input
            id="datamind-username"
            value={username}
            onChange={(event) => {
              setUsername(event.target.value);
              setError(null);
            }}
            onFocus={() => setFocus("username")}
            onBlur={() => setFocus("idle")}
            className="input"
            placeholder="例如 default 或 nora@datamind.local"
            autoComplete="username"
            aria-invalid={!!error}
            aria-describedby={error ? "login-error" : undefined}
          />
          <label className="label mt-4" htmlFor="datamind-password">密码</label>
          <input
            id="datamind-password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
              setError(null);
            }}
            onFocus={() => setFocus("password")}
            onBlur={() => setFocus("idle")}
            className="input"
            placeholder="首次登录会创建本地用户"
            type="password"
            autoComplete="current-password"
            aria-invalid={!!error}
            aria-describedby={error ? "login-error" : undefined}
          />
          {(error || notice) && (
            <div id="login-error">
              <Alert tone={error ? "error" : "info"}>{error ?? notice}</Alert>
            </div>
          )}
          <button type="submit" disabled={busy} className="login-button mt-6">
            <span>{busy ? "登录中" : "Log in"}</span>
            {busy ? <Loader2 className="animate-spin" size={18} /> : <span aria-hidden="true">→</span>}
          </button>
        </form>
      </section>
    </main>
  );
}

function LoginCharacter({
  tone,
  focus,
  pointer,
  delay = false,
  half = false,
  small = false,
}: {
  tone: "purple" | "dark" | "orange" | "yellow";
  focus: "idle" | "username" | "password";
  pointer: { x: number; y: number };
  delay?: boolean;
  half?: boolean;
  small?: boolean;
}) {
  const eyeX = focus === "password" ? -7 : focus === "username" ? 6 : pointer.x * 10;
  const eyeY = focus === "password" ? -2 : pointer.y * 5;
  return (
    <div
      className={`login-character ${tone} ${half ? "half" : ""} ${small ? "small" : ""} ${delay ? "delay" : ""} ${focus}`}
      style={{ transform: `rotate(${pointer.x * 7}deg) translateY(${pointer.y * 8}px)` }}
    >
      <span className="eye left" style={{ transform: `translate(${eyeX}px, ${eyeY}px)` }} />
      <span className="eye right" style={{ transform: `translate(${eyeX}px, ${eyeY}px)` }} />
      <span className="mouth" />
    </div>
  );
}

function Sidebar({
  page,
  user,
  assistantActiveRuns,
  onPageChange,
  onLogout,
}: {
  page: Page;
  user: AuthUser;
  assistantActiveRuns: number;
  onPageChange: (page: Page) => void;
  onLogout: () => void;
}) {
  const items: { page: Page; icon: React.ReactNode }[] = [
    { page: "首页", icon: <Home size={16} /> },
    { page: "数据集", icon: <Database size={16} /> },
    { page: "分析任务", icon: <Play size={16} /> },
    { page: "报告", icon: <FileText size={16} /> },
    { page: "Kimi", icon: <Sparkles size={16} /> },
  ];
  return (
    <aside className="fixed inset-x-0 bottom-0 z-20 flex h-[76px] w-full items-center border-t border-line bg-white/95 px-2 py-2 shadow-[0_-12px_32px_rgba(15,23,42,0.08)] backdrop-blur md:inset-y-0 md:left-0 md:right-auto md:h-auto md:w-[236px] md:flex-col md:items-stretch md:border-r md:border-t-0 md:px-8 md:py-8 md:shadow-[18px_0_44px_rgba(15,23,42,0.05)]">
      <div className="mb-14 hidden items-center gap-3 md:flex">
        <div className="grid h-10 w-10 place-items-center rounded-2xl bg-slate-950 text-xs font-black text-white shadow-[0_14px_28px_rgba(15,23,42,0.2)]">
          DM
        </div>
        <strong className="text-lg tracking-tight">DataMind</strong>
      </div>
      <nav className="flex min-w-0 flex-1 items-center justify-around gap-1 md:block md:space-y-3">
        {items.map((item) => (
          <button
            key={item.page}
            onClick={() => onPageChange(item.page)}
            className={`flex h-14 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 text-center text-[10px] font-black transition md:h-12 md:w-full md:flex-row md:justify-start md:gap-3 md:rounded-xl md:px-5 md:text-left md:text-sm ${
              page === item.page
                ? "bg-acid text-black shadow-[0_14px_24px_rgba(200,251,79,0.34)]"
                : "bg-transparent text-slate-700 hover:bg-slate-100 hover:text-slate-950"
            }`}
          >
            {item.icon}
            <span className="relative">
              {item.page}
              {item.page === "Kimi" && assistantActiveRuns > 0 && (
                <i className="absolute -right-5 -top-2 grid min-h-4 min-w-4 place-items-center rounded-full bg-emerald-600 px-1 text-[9px] font-black not-italic text-white">
                  {assistantActiveRuns}
                </i>
              )}
            </span>
          </button>
        ))}
        <button
          type="button"
          aria-label="Log Out"
          onClick={onLogout}
          className="flex h-14 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 text-[10px] font-black text-slate-600 transition hover:bg-slate-100 hover:text-slate-950 md:hidden"
        >
          <LogOut size={16} />
          退出
        </button>
      </nav>
      <div className="mt-auto hidden w-full min-w-0 space-y-4 overflow-hidden text-center md:block">
        <div className="mx-auto grid h-10 w-10 place-items-center rounded-full border border-line bg-slate-100 text-sm font-black">
          {user.display_name.slice(0, 1).toUpperCase()}
        </div>
        <div
          className="w-full min-w-0 truncate px-1 text-sm font-black"
          data-testid="sidebar-account-name"
          title={user.display_name}
        >
          {user.display_name}
        </div>
        <button
          type="button"
          onClick={onLogout}
          className="mx-auto flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-bold text-slate-600 transition hover:bg-slate-100 hover:text-slate-950"
        >
          <LogOut size={15} /> Log Out
        </button>
      </div>
    </aside>
  );
}

function Topbar({ user }: { user: AuthUser }) {
  return (
    <header className="mb-10 flex items-start justify-between">
      <h1 className="text-3xl font-black tracking-tight">DataMind</h1>
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-full border border-[#eadfd4] bg-[#f4eadf] text-xs font-black shadow-[0_10px_20px_rgba(15,23,42,0.08)]">
          {user.display_name.slice(0, 1).toUpperCase()}
        </span>
        <div>
          <b className="block text-sm">{user.display_name}</b>
          <small className="text-xs text-slate-500">DataMind User</small>
        </div>
      </div>
    </header>
  );
}

function FloatingTaskProgress({
  task,
  activeCount,
  onOpen,
}: {
  task: ActiveTask;
  activeCount: number;
  onOpen: () => void;
}) {
  const progress = Math.max(0, Math.min(task.progress, 100));
  const Icon = task.kind === "cleaning" ? Database : task.kind === "assistant" ? Sparkles : Loader2;
  const motionClass = task.kind === "analysis" ? "animate-spin" : "animate-pulse";
  const accessibleLabel = task.kind === "analysis"
    ? `查看运行中的分析：${task.title}，${progress}%`
    : task.kind === "cleaning"
      ? `查看后台清洗进度，${progress}%`
      : `查看后台助手进度，${progress}%`;
  return (
    <button
      type="button"
      className="floating-task-progress"
      onClick={onOpen}
      aria-label={accessibleLabel}
      title={`前往${task.page}`}
    >
      <span className="floating-task-icon"><Icon className={motionClass} size={18} /></span>
      <span className="floating-task-copy">
        <span>
          {activeCount > 1 ? `${activeCount} 个任务运行中` : task.stage}
          <b>{progress}%</b>
        </span>
        <strong>{task.title}</strong>
      </span>
      <ArrowRight className="floating-task-arrow" size={17} />
      <span className="floating-task-track" aria-hidden="true">
        <i style={{ width: `${Math.max(progress, 3)}%` }} />
      </span>
    </button>
  );
}

function cleaningStageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: "等待清洗",
    cleaning_bootstrap: "准备数据范围",
    cleaning_decision: "选择清洗策略",
    cleaning_execution: "执行清洗",
    cleaning_validation: "验证清洗质量",
    cleaning_repair: "修复清洗规则",
    cleaning_commit: "保存清洗版本",
  };
  return labels[stage] ?? "自主清洗运行中";
}

function assistantStageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: "等待 Kimi 处理",
    retrieve_context: "检索 DataMind 证据",
    decide_tools: "规划工具调用",
    execute_tool: "执行 DataMind 工具",
    wait_analysis: "等待分析完成",
    compose_answer: "整理最终回答",
  };
  return labels[stage] ?? "Kimi 正在后台运行";
}

function Dashboard({
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
  const readiness = datasets.length ? Math.round((cleanedDatasets.length / datasets.length) * 100) : 0;
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

function DatasetsPage({
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



function AnalysisPage({
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

function DynamicAgentPlan({ job }: { job: AnalysisJob | null }) {
  const workflowSteps = useMemo(() => deriveAgentWorkflowViews(job), [job]);
  const statusByKey = new Map(workflowSteps.map((step) => [step.key, step.status]));
  const planSteps = job?.agent_mode === "loop"
    ? [
        { key: "analyze", label: "按需分析", workflowKeys: ["sql", "python"] },
        { key: "visualize", label: "可视化", workflowKeys: ["visualization"] },
        { key: "report", label: "报告", workflowKeys: ["reviewer", "report"] },
      ]
    : AGENT_PLAN_STEPS;
  return (
    <div>
      <div className="mb-2 text-xs font-black uppercase tracking-wide text-slate-500">智能体计划</div>
      <div className="flex flex-wrap gap-2">
        {planSteps.map((plan) => {
          const status = combinedWorkflowStatus(
            plan.workflowKeys.map((key) => statusByKey.get(key as AgentWorkflowStepKey) ?? "waiting"),
          );
          return (
            <span key={plan.key} className={`agent-plan-pill ${agentStatusClass(status)}`}>
              <AgentStatusIcon status={status} size="sm" />
              {plan.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function RealtimeWorkflowPanel({ job }: { job: AnalysisJob | null }) {
  const workflowSteps = useMemo(() => deriveAgentWorkflowViews(job), [job]);
  return (
    <div className="analysis-workflow-panel">
      <div className="analysis-workflow-rail">
        {workflowSteps.map((step, index) => (
          <React.Fragment key={step.key}>
            {index > 0 && <span className="hidden text-slate-300 md:inline">→</span>}
            <div className={`workflow-node ${agentStatusClass(step.status)}`}>
              <AgentStatusIcon status={step.status} />
              <span>{step.label}</span>
            </div>
          </React.Fragment>
        ))}
      </div>
      <div className="analysis-workflow-details">
        {workflowSteps.map((step) => (
          <details key={step.key} className="analysis-workflow-detail">
            <summary>
              <span className="mr-2 inline-flex align-middle">
                <AgentStatusIcon status={step.status} size="sm" />
              </span>
              {step.label}详情
            </summary>
            <p className="analysis-workflow-detail-copy">{step.detail}</p>
            <div className="mt-2 space-y-1">
              {step.events.map((event, index) => (
                <div key={`${event.created_at}-${index}`} className="analysis-workflow-event">
                  {formatTime(event.created_at)} · {translateWorkflowEventMessage(event)}
                </div>
              ))}
              {!step.events.length && <div className="text-xs text-slate-400">等待事件...</div>}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}

function AnalysisJobStatusPanel({
  job,
  onCancel,
  onRetry,
}: {
  job: AnalysisJob;
  onCancel: () => Promise<void>;
  onRetry: () => Promise<void>;
}) {
  const active = isActiveAnalysisJob(job);
  const logRef = useRef<HTMLDivElement | null>(null);
  const logEntries = useMemo(() => buildWorkflowLogEntries(job), [job]);
  useEffect(() => {
    const element = logRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [logEntries.length]);
  return (
    <div className="analysis-job-status-panel">
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-black text-slate-950">
            {jobStatusLabel(job.status)} · {jobStageLabel(job.current_stage)}
          </div>
          <div className="mt-1 break-words text-xs font-bold text-slate-500 [overflow-wrap:anywhere]">
            {job.question} · {formatTime(job.updated_at)}
          </div>
        </div>
        <div className="flex gap-2">
          {active && (
            <button type="button" className="small-button" onClick={() => void onCancel()}>
              取消
            </button>
          )}
          {!active && job.status !== "completed" && (
            <button type="button" className="small-button" onClick={() => void onRetry()}>
              重试
            </button>
          )}
        </div>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-emerald-500 transition-all"
          style={{ width: `${Math.max(0, Math.min(job.progress, 100))}%` }}
        />
      </div>
      <div className="mt-2 text-xs font-bold text-slate-500">{job.progress}%</div>
      {!!job.events.length && (
        <details className="mt-4 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2" open={active}>
          <summary className="cursor-pointer text-sm font-black text-slate-700">
            <span>运行日志</span>
            <span className="ml-1 text-xs font-bold text-slate-400">· {logEntries.length} 条</span>
          </summary>
          <div ref={logRef} className="mt-2 max-h-56 space-y-2 overflow-auto rounded-xl border border-slate-200 bg-slate-950 p-3">
            {logEntries.map((entry, index) => (
              <div key={`${entry.createdAt}-${index}`} className={`workflow-log-line ${entry.kind}`}>
                <span>{entry.icon}</span>
                <span>{entry.text}</span>
                <time>{formatTime(entry.createdAt)}</time>
              </div>
            ))}
          </div>
        </details>
      )}
      {!!job.events.length && (
        <details className="mt-3 rounded-xl border border-line bg-slate-50 px-4 py-3">
          <summary className="cursor-pointer text-sm font-black text-slate-800">诊断事件（高级）</summary>
          <div className="mt-3 grid gap-2">
            {job.events.map((event, index) => (
              <div key={`${event.created_at}-${index}`} className="break-words rounded-lg bg-white px-3 py-2 text-xs leading-5 text-slate-600 [overflow-wrap:anywhere]">
                <b>{jobStageLabel(event.stage)}</b> · {translateWorkflowEventMessage(event)}
                <span className="ml-2 text-slate-400">{event.progress}%</span>
              </div>
            ))}
          </div>
        </details>
      )}
      {job.error && <Alert tone="error">{job.error}</Alert>}
    </div>
  );
}

function AgentStatusIcon({ status, size = "md" }: { status: WorkflowNodeStatus; size?: "sm" | "md" }) {
  const iconSize = size === "sm" ? 12 : 14;
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1">
        <span className="workflow-running-dot">◉</span>
        <Loader2 className="animate-spin" size={iconSize} />
      </span>
    );
  }
  if (status === "completed") return <span className="workflow-complete-icon">✔</span>;
  if (status === "failed") return <span className="workflow-failed-icon">✖</span>;
  return <span className="workflow-waiting-icon">○</span>;
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

function TextAnalysisPanel({ results }: { results: TextAnalysisResult[] }) {
  if (!results.length) return null;
  return (
    <section className="mt-5 rounded-xl border border-emerald-100 bg-emerald-50/40 p-4">
      <h4 className="font-black text-slate-950">文本分析工具箱</h4>
      <div className="mt-3 grid gap-3">
        {results.map((result, index) => {
          const topKeywords = arrayOfRecords(result.summary.top_keywords);
          const groups = arrayOfRecords(result.summary.groups);
          return (
            <div key={`${result.text_column}-${index}`} className="rounded-xl border border-line bg-white p-4 text-sm leading-6">
              <div className="flex flex-wrap items-center gap-2">
                <b>{result.text_column}</b>
                {result.group_column && (
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-black text-slate-600">
                    按 {result.group_column} 分组
                  </span>
                )}
                <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-black text-emerald-700">
                  {result.task}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
                <MetricPill label="可分析文本" value={String(result.summary.non_empty_count ?? 0)} />
                <MetricPill label="空文本" value={String(result.summary.empty_count ?? 0)} />
                <MetricPill label="平均长度" value={String(result.summary.avg_length ?? 0)} />
                <MetricPill label="最长文本" value={String(result.summary.max_length ?? 0)} />
              </div>
              {!!result.insights.length && (
                <ul className="mt-3 list-disc space-y-1 pl-5 text-slate-700">
                  {result.insights.map((insight) => (
                    <li key={insight}>{insight}</li>
                  ))}
                </ul>
              )}
              {!!topKeywords.length && (
                <div className="mt-3">
                  <p className="text-xs font-black uppercase tracking-wide text-slate-500">高频关键词</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {topKeywords.slice(0, 12).map((keyword) => (
                      <span key={String(keyword.keyword)} className="rounded-full bg-slate-100 px-2 py-1 text-xs font-bold text-slate-700">
                        {String(keyword.keyword)} · {String(keyword.count)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {!!groups.length && (
                <details className="mt-3 rounded-lg border border-line bg-slate-50 px-3 py-2">
                  <summary className="cursor-pointer font-black">查看分组文本画像</summary>
                  <DataTable rows={groups} emptyText="暂无分组结果。" />
                </details>
              )}
              <ChartList charts={result.charts} />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-slate-50 px-3 py-2">
      <p className="text-xs font-bold text-slate-500">{label}</p>
      <p className="text-lg font-black text-slate-950">{value}</p>
    </div>
  );
}

function ReportsPage({
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

function MultimodalContextPanel({ inputs, compact = false }: { inputs: MultimodalInput[]; compact?: boolean }) {
  if (!inputs.length) return null;
  return (
    <section className={compact ? "mt-4" : ""}>
      <h3 className={compact ? "text-sm font-black text-slate-700" : "section-title"}>多模态上下文</h3>
      <div className="mt-2 grid gap-2">
        {inputs.map((item, index) => (
          <div key={`${item.title}-${index}`} className="rounded-xl border border-line bg-white px-4 py-3 text-sm leading-6">
            <div className="flex flex-wrap items-center gap-2">
              <b>{item.title}</b>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-black text-slate-600">
                {multimodalKindLabel(item.kind)}
              </span>
              {item.media_type && (
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-black text-slate-600">
                  {item.media_type}
                </span>
              )}
              <span className={`rounded-full px-2 py-0.5 text-xs font-black ${multimodalStatusClass(item.processing_status)}`}>
                {multimodalStatusLabel(item)}
              </span>
            </div>
            <p className="mt-2 text-slate-600">{item.description}</p>
            {item.text_excerpt && (
              <details className="mt-3 rounded-lg border border-line bg-slate-50 px-3 py-2">
                <summary className="cursor-pointer font-black">查看 PDF 抽取文本</summary>
                <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-700">{item.text_excerpt}</pre>
              </details>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function ReportVersionCompare({ left, right }: { left: Report; right: Report }) {
  const leftReport = structuredReportFromMetadata(left.metadata);
  const rightReport = structuredReportFromMetadata(right.metadata);
  const rows = [
    {
      项目: "标题",
      对比版本: left.title,
      当前版本: right.title,
      状态: left.title === right.title ? "相同" : "变化",
    },
    {
      项目: "问题",
      对比版本: String(left.metadata?.question ?? ""),
      当前版本: String(right.metadata?.question ?? ""),
      状态: String(left.metadata?.question ?? "") === String(right.metadata?.question ?? "") ? "相同" : "变化",
    },
    {
      项目: "Executive Summary",
      对比版本: leftReport?.executive_summary ?? left.markdown.slice(0, 180),
      当前版本: rightReport?.executive_summary ?? right.markdown.slice(0, 180),
      状态: (leftReport?.executive_summary ?? left.markdown) === (rightReport?.executive_summary ?? right.markdown) ? "相同" : "变化",
    },
    {
      项目: "Key Findings",
      对比版本: String(leftReport?.key_findings?.length ?? 0),
      当前版本: String(rightReport?.key_findings?.length ?? 0),
      状态: (leftReport?.key_findings?.length ?? 0) === (rightReport?.key_findings?.length ?? 0) ? "相同" : "变化",
    },
    {
      项目: "Validation Issues",
      对比版本: String(leftReport?.validation_issues?.length ?? 0),
      当前版本: String(rightReport?.validation_issues?.length ?? 0),
      状态: (leftReport?.validation_issues?.length ?? 0) === (rightReport?.validation_issues?.length ?? 0) ? "相同" : "变化",
    },
    {
      项目: "Recommended Next Steps",
      对比版本: String(leftReport?.recommended_next_steps?.length ?? 0),
      当前版本: String(rightReport?.recommended_next_steps?.length ?? 0),
      状态: (leftReport?.recommended_next_steps?.length ?? 0) === (rightReport?.recommended_next_steps?.length ?? 0) ? "相同" : "变化",
    },
  ];
  return (
    <section className="report-toolbar mb-4 rounded-xl border border-amber-200 bg-amber-50 p-4">
      <h4 className="section-heading">版本对比</h4>
      <p className="mb-3 text-sm font-bold text-slate-600">
        v{left.version ?? 1} 对比 v{right.version ?? 1}
      </p>
      <DataTable rows={rows} emptyText="暂无可对比字段。" />
    </section>
  );
}

function Metric({ label, value, caption, icon }: { label: string; value: number | string; caption: string; icon?: React.ReactNode }) {
  return (
    <div className="metric-card">
      <div className="metric-card-heading">
        <span>{label}</span>
        {icon && <i>{icon}</i>}
      </div>
      <strong className="mt-4 block text-3xl font-black tracking-tight text-slate-950">{value}</strong>
      <small className="mt-2 block text-sm text-slate-500">{caption}</small>
    </div>
  );
}

function DataTable({ rows, emptyText }: { rows: Record<string, unknown>[]; emptyText: string }) {
  const columns = useMemo(() => Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 8), [rows]);
  if (!rows.length) return <Alert>{emptyText}</Alert>;
  return (
    <div className="max-h-[360px] overflow-auto rounded-xl border border-line bg-white shadow-[0_10px_26px_rgba(15,23,42,0.05)]">
      <table className="w-full border-collapse text-left text-sm text-slate-800">
        <thead className="sticky top-0 bg-slate-100 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {columns.map((column) => (
              <th key={column} className="border-b border-line px-3 py-3 font-black">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 80).map((row, index) => (
            <tr key={index} className="border-b border-slate-100 odd:bg-white even:bg-slate-50/70">
              {columns.map((column) => (
                <td key={column} className="max-w-[280px] truncate px-3 py-3">
                  {String(row[column] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartList({ charts }: { charts: Chart[] }) {
  if (!charts.length) return null;
  return (
    <div className="mt-4 grid min-w-0 gap-4">
      {charts.map((chart, index) => <ChartCard key={`${chart.title}-${index}`} chart={chart} />)}
    </div>
  );
}

function ChartCard({ chart }: { chart: Chart }) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const displayScopeMessage = chartDisplayScopeMessage(chart);
  return (
    <div ref={cardRef} className="chart-card">
      <div className="chart-card-header">
        <div className="min-w-0">
          <p className="chart-card-eyebrow">数据可视化</p>
          <h4 className="chart-card-title">{chart.title}</h4>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2 report-toolbar">
          <span className="chart-type-label">{chartTypeLabel(chart.chart_type)}</span>
          <button type="button" className="small-button h-8 px-2.5" title="导出 SVG" onClick={() => void exportChart(cardRef.current, chart.title, "svg")}><Download size={14} />SVG</button>
          <button type="button" className="small-button h-8 px-2.5" title="导出高清 PNG" onClick={() => void exportChart(cardRef.current, chart.title, "png")}><Download size={14} />PNG</button>
        </div>
      </div>
      <div className="chart-visualization"><ChartVisualization chart={chart} /></div>
      {displayScopeMessage && <p className="chart-explanation">{displayScopeMessage}</p>}
      {chart.explanation && <p className="chart-explanation">{chart.explanation}</p>}
      <details className="chart-data-disclosure"><summary>查看图表数据</summary><DataTable rows={chart.data} emptyText="图表没有数据。" /></details>
    </div>
  );
}

function StructuredReportPreview({ report }: { report: StructuredReport }) {
  const textAnalysis = textAnalysisFromUnknown(report.python_results?.text_analysis);
  return (
    <div className="mt-5 space-y-5">
      <section className="rounded-xl border border-sky-200 bg-sky-50 p-5 text-sm leading-7 text-slate-950">{report.executive_summary}</section>
      <AnalysisReliabilityPanel
        contract={report.analysis_contract}
        verification={report.statistical_verification}
        lineage={report.analysis_lineage}
      />
      {report.analysis_context && (
        <section className="rounded-xl border border-line bg-white p-4 text-sm leading-6 text-slate-700">
          <p className="font-black text-slate-950">分析上下文</p>
          <p className="mt-2">{report.analysis_context}</p>
        </section>
      )}
      <TextAnalysisPanel results={textAnalysis} />
      {!!report.key_findings?.length && (
        <section>
          <h4 className="section-heading">核心发现</h4>
          <div className="grid gap-3">
            {report.key_findings.map((finding) => (
              <div key={finding.title} className="rounded-xl border border-line bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)]">
                <p className="font-black">{finding.title}</p>
                <p className="mt-2 text-sm leading-6">{finding.content}</p>
                {(finding.evidence || finding.business_impact || finding.recommended_action) && (
                  <div className="mt-3 grid gap-2 text-xs leading-5 text-slate-600">
                    {finding.evidence && <span>证据：{finding.evidence}</span>}
                    {finding.business_impact && <span>影响：{finding.business_impact}</span>}
                    {finding.recommended_action && <span>建议：{finding.recommended_action}</span>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
      <ChartList charts={report.charts ?? []} />
      {!!report.recommended_next_steps?.length && (
        <section>
          <h4 className="section-heading">建议下一步</h4>
          <div className="grid gap-2">
            {report.recommended_next_steps.map((step, index) => (
              <div key={`${step}-${index}`} className="rounded-xl border border-line bg-white px-4 py-3 text-sm">
                {step}
              </div>
            ))}
          </div>
        </section>
      )}
      {!!report.sql_results?.length && (
        <section>
          <h4 className="section-heading">SQL 结果</h4>
          <DataTable rows={report.sql_results} emptyText="没有 SQL 结果。" />
        </section>
      )}
      {!!report.data_gaps?.length && (
        <section>
          <h4 className="section-heading">数据缺口</h4>
          <div className="grid gap-2">
            {report.data_gaps.map((gap) => (
              <Alert key={gap}>{gap}</Alert>
            ))}
          </div>
        </section>
      )}
      {!!report.validation_issues?.length && (
        <section>
          <h4 className="section-heading">校验问题</h4>
          <div className="grid gap-2">
            {report.validation_issues.map((issue, index) => (
              <div key={`${issue.finding_ref}-${index}`} className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6">
                <b>{issue.severity}</b> · {issue.finding_ref}: {issue.issue}
                {issue.suggestion && <p className="text-slate-600">{issue.suggestion}</p>}
              </div>
            ))}
          </div>
        </section>
      )}
      {!!report.analysis_trace?.length && (
        <section>
          <h4 className="section-heading">分析轨迹</h4>
          <div className="grid gap-2">
            {report.analysis_trace.map((round) => (
              <TraceRoundCard key={round.round_number} round={round} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function TraceRoundCard({ round }: { round: NonNullable<StructuredReport["analysis_trace"]>[number] }) {
  const execution = round.execution_result ?? {};
  const fanoutMode = String(execution.fanout_mode ?? "serial");
  const isFanout = fanoutMode.includes("fanout");
  return (
    <div className="rounded-xl border border-line bg-white px-4 py-3 text-sm leading-6">
      <div className="flex flex-wrap items-center gap-2">
        <p className="font-black">Round {round.round_number}</p>
        <span className={`rounded-full px-2 py-0.5 text-xs font-black ${round.validation_status === "warning" ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-700"}`}>
          {round.validation_status ?? "passed"}
        </span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-black ${isFanout ? "bg-violet-50 text-violet-700" : "bg-slate-100 text-slate-600"}`}>
          {isFanout ? "并行探索" : "串行基础"}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-5">
        <MetricPill label="路线" value={round.plan?.route ?? "analysis"} />
        <MetricPill label="SQL 行数" value={metricValue(execution.sql_row_count)} />
        <MetricPill label="图表" value={metricValue(execution.chart_count)} />
        <MetricPill label="文本分析" value={metricValue(execution.text_analysis_count)} />
        <MetricPill label="Python" value={String(execution.python_source ?? "-")} />
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-xs font-bold text-slate-500">
        {round.plan?.metric_column && <span>指标：{round.plan.metric_column}</span>}
        {round.plan?.category_column && <span>维度：{round.plan.category_column}</span>}
        {round.plan?.time_column && <span>时间：{round.plan.time_column}</span>}
        {Boolean(execution.fanout_group) && <span>分支组：{String(execution.fanout_group)}</span>}
      </div>
      <p className="mt-3 font-bold text-slate-950">{round.hypothesis?.statement}</p>
      <p className="text-slate-600">{round.reflection?.insight_text}</p>
      {Boolean(execution.python_execution_error) && (
        <p className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700">
          Python 回退：{String(execution.python_execution_error)}
        </p>
      )}
      {executionIssues(execution, "plan_validation_issues").map((issue, index) => (
        <p key={`${round.round_number}-plan-issue-${index}`} className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
          Plan Harness：{issue.finding_ref ? `${issue.finding_ref} · ` : ""}
          {issue.issue}
        </p>
      ))}
    </div>
  );
}

function ChartVisualization({ chart }: { chart: Chart }) {
  if (!chart.data.length) return <Alert>图表没有数据。</Alert>;
  if (chart.chart_type === "pie") return <PieChartSvg chart={chart} />;
  if (chart.chart_type === "line") return <CartesianChartSvg chart={chart} mode="line" />;
  if (chart.chart_type === "histogram") return <HistogramChartSvg chart={chart} />;
  if (chart.chart_type === "box_plot") return <BoxPlotSvg chart={chart} />;
  if (chart.chart_type === "correlation_heatmap") return <HeatmapSvg chart={chart} />;
  return <CartesianChartSvg chart={chart} mode="bar" />;
}

function CartesianChartSvg({ chart, mode }: { chart: Chart; mode: "bar" | "line" }) {
  const chartId = useId().replace(/:/g, "");
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "x");
  const cartesianKeys = Object.keys(chart.data[0] ?? {});
  const yKey = String(chart.spec.y ?? cartesianKeys[cartesianKeys.length - 1] ?? "y");
  const visibleRows = mode === "bar" && chart.data.length > 24
    ? [...chart.data].sort((leftRow, rightRow) => numberValue(rightRow[yKey]) - numberValue(leftRow[yKey]))
    : chart.data;
  const points = visibleRows
    .slice(0, 24)
    .map((row) => ({ label: String(row[xKey] ?? ""), value: numberValue(row[yKey]) }))
    .filter((point) => Number.isFinite(point.value));
  if (!points.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  const width = 760;
  const height = 300;
  const left = 64;
  const right = 28;
  const top = 42;
  const bottom = 58;
  const max = Math.max(...points.map((point) => point.value), 1);
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const step = innerWidth / Math.max(points.length, 1);
  const ticks = axisTicks(max);
  const coordinates = points.map((point, index) => ({
    ...point,
    x: left + step * index + step / 2,
    y: height - bottom - (point.value / max) * innerHeight,
  }));
  const linePath = coordinates.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const areaPath = `${linePath} L ${coordinates[coordinates.length - 1].x} ${height - bottom} L ${coordinates[0].x} ${height - bottom} Z`;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={chart.title}>
      <defs>
        <linearGradient id={`${chartId}-area`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={CHART_LINE_COLOR} stopOpacity="0.24" />
          <stop offset="100%" stopColor={CHART_LINE_COLOR} stopOpacity="0" />
        </linearGradient>
        <filter id={`${chartId}-shadow`} x="-20%" y="-20%" width="140%" height="160%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" floodColor="#334155" floodOpacity="0.14" />
        </filter>
      </defs>
      <text x={left} y={22} fontSize="11" fontWeight="700" fill="#64748b">{yKey}</text>
      {ticks.map((tick) => {
        const y = height - bottom - (tick / max) * innerHeight;
        return (
          <g key={`tick-${tick}`}>
            <line x1={left} y1={y} x2={width - right} y2={y} stroke="#dbe4ee" strokeDasharray="4 6" />
            <text x={left - 12} y={y + 4} textAnchor="end" fontSize="10" fontWeight="600" fill="#64748b">
              {formatAxisValue(tick)}
            </text>
          </g>
        );
      })}
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} stroke="#bac8d8" />
      {mode === "bar"
        ? coordinates.map((point, index) => {
            const barWidth = Math.min(Math.max(step * 0.58, 8), 46);
            const barHeight = (point.value / max) * innerHeight;
            const x = point.x - barWidth / 2;
            const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
            return (
              <rect key={`${point.label}-${index}`} x={x} y={point.y} width={barWidth} height={barHeight} rx={7} fill={color} fillOpacity={0.94} filter={`url(#${chartId}-shadow)`}>
                <title>{`${point.label}: ${formatAxisValue(point.value)}`}</title>
              </rect>
            );
          })
        : (
            <>
              <path d={areaPath} fill={`url(#${chartId}-area)`} />
              <path d={linePath} fill="none" stroke={CHART_LINE_COLOR} strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round" />
              {coordinates.map((point, index) => (
                <circle key={`${point.label}-${index}`} cx={point.x} cy={point.y} r={4.5} fill="white" stroke={CHART_LINE_COLOR} strokeWidth={3}>
                  <title>{`${point.label}: ${formatAxisValue(point.value)}`}</title>
                </circle>
              ))}
            </>
          )}
      {coordinates.map((point, index) => (
        <text key={`${point.label}-label-${index}`} x={point.x} y={height - 22} textAnchor="middle" fontSize="10" fontWeight="600" fill="#526174">
          {shortLabel(point.label)}
        </text>
      ))}
    </svg>
  );
}

function PieChartSvg({ chart }: { chart: Chart }) {
  const nameKey = String(chart.spec.names ?? Object.keys(chart.data[0] ?? {})[0] ?? "name");
  const pieKeys = Object.keys(chart.data[0] ?? {});
  const valueKey = String(chart.spec.values ?? pieKeys[pieKeys.length - 1] ?? "value");
  const slices = chart.data
    .slice(0, 8)
    .map((row) => ({ label: String(row[nameKey] ?? ""), value: Math.max(numberValue(row[valueKey]), 0) }))
    .filter((slice) => slice.value > 0);
  const total = slices.reduce((sum, slice) => sum + slice.value, 0);
  if (!total) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  let current = 0;
  return (
    <div className="chart-pie-layout">
      <svg viewBox="0 0 220 220" className="chart-pie-svg" role="img" aria-label={chart.title}>
        {slices.map((slice, index) => {
          const start = current;
          const end = current + (slice.value / total) * Math.PI * 2;
          current = end;
          return (
            <path key={slice.label} d={arcPath(110, 110, 82, start, end)} fill={CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]} stroke="white" strokeWidth="3" strokeLinejoin="round">
              <title>{`${slice.label}: ${formatAxisValue(slice.value)} (${((slice.value / total) * 100).toFixed(1)}%)`}</title>
            </path>
          );
        })}
        <circle cx="110" cy="110" r="48" fill="white" />
        <text x="110" y="104" textAnchor="middle" fontSize="11" fontWeight="700" fill="#64748b">合计</text>
        <text x="110" y="124" textAnchor="middle" fontSize="16" fontWeight="800" fill="#0f172a">{formatAxisValue(total)}</text>
      </svg>
      <div className="chart-legend">
        {slices.map((slice, index) => (
          <div key={slice.label} className="chart-legend-row">
            <span className="flex min-w-0 items-center gap-2.5">
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length] }} />
              <span className="truncate font-bold text-slate-700">{slice.label}</span>
            </span>
            <span className="flex shrink-0 items-baseline gap-2">
              <b className="text-slate-950">{formatAxisValue(slice.value)}</b>
              <span className="text-xs font-bold text-slate-500">{((slice.value / total) * 100).toFixed(1)}%</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HistogramChartSvg({ chart }: { chart: Chart }) {
  if (chart.spec.y) return <CartesianChartSvg chart={chart} mode="bar" />;
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "value");
  const values = chart.data.map((row) => numberValue(row[xKey])).filter(Number.isFinite);
  if (!values.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bucketCount = 10;
  const buckets = Array.from({ length: bucketCount }, (_, index) => ({ label: `${index + 1}`, value: 0 }));
  values.forEach((value) => {
    const ratio = max === min ? 0 : (value - min) / (max - min);
    buckets[Math.min(bucketCount - 1, Math.floor(ratio * bucketCount))].value += 1;
  });
  return <CartesianChartSvg chart={{ ...chart, chart_type: "bar", spec: { x: "label", y: "value" }, data: buckets }} mode="bar" />;
}

function BoxPlotSvg({ chart }: { chart: Chart }) {
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "category");
  const boxKeys = Object.keys(chart.data[0] ?? {});
  const yKey = String(chart.spec.y ?? boxKeys[boxKeys.length - 1] ?? "value");
  const groups = new Map<string, number[]>();
  chart.data.forEach((row) => {
    const label = String(row[xKey] ?? "未分组");
    const value = numberValue(row[yKey]);
    if (Number.isFinite(value)) groups.set(label, [...(groups.get(label) ?? []), value]);
  });
  const summaries = Array.from(groups.entries()).slice(0, 8).map(([label, values]) => ({ label, ...quartiles(values) }));
  if (!summaries.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  const width = 760;
  const height = 300;
  const left = 64;
  const right = 28;
  const top = 42;
  const bottom = 58;
  const max = Math.max(...summaries.map((item) => item.max), 1);
  const scaleY = (value: number) => height - bottom - (value / max) * (height - top - bottom);
  const step = (width - left - right) / summaries.length;
  const ticks = axisTicks(max);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={chart.title}>
      <text x={left} y={22} fontSize="11" fontWeight="700" fill="#64748b">{yKey}</text>
      {ticks.map((tick) => {
        const y = scaleY(tick);
        return (
          <g key={`box-tick-${tick}`}>
            <line x1={left} y1={y} x2={width - right} y2={y} stroke="#dbe4ee" strokeDasharray="4 6" />
            <text x={left - 12} y={y + 4} textAnchor="end" fontSize="10" fontWeight="600" fill="#64748b">
              {formatAxisValue(tick)}
            </text>
          </g>
        );
      })}
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} stroke="#bac8d8" />
      {summaries.map((item, index) => {
        const x = left + step * index + step / 2;
        const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
        return (
          <g key={item.label}>
            <line x1={x} x2={x} y1={scaleY(item.min)} y2={scaleY(item.max)} stroke={color} strokeWidth={2.5} />
            <line x1={x - 12} x2={x + 12} y1={scaleY(item.min)} y2={scaleY(item.min)} stroke={color} strokeWidth={2.5} />
            <line x1={x - 12} x2={x + 12} y1={scaleY(item.max)} y2={scaleY(item.max)} stroke={color} strokeWidth={2.5} />
            <rect x={x - 22} y={scaleY(item.q3)} width={44} height={Math.max(scaleY(item.q1) - scaleY(item.q3), 3)} rx={6} fill={color} fillOpacity={0.28} stroke={color} strokeWidth={1.8}>
              <title>{`${item.label} · 最小 ${formatAxisValue(item.min)} · 中位数 ${formatAxisValue(item.median)} · 最大 ${formatAxisValue(item.max)}`}</title>
            </rect>
            <line x1={x - 22} x2={x + 22} y1={scaleY(item.median)} y2={scaleY(item.median)} stroke="#0f172a" strokeWidth={3} />
            <text x={x} y={height - 22} textAnchor="middle" fontSize="10" fontWeight="600" fill="#526174">{shortLabel(item.label)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function HeatmapSvg({ chart }: { chart: Chart }) {
  const labels = Array.from(new Set(chart.data.flatMap((row) => [String(row.source ?? ""), String(row.target ?? "")]))).filter(Boolean).slice(0, 10);
  if (!labels.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数据。" />;
  const cell = 44;
  const pad = 90;
  const size = pad + labels.length * cell + 16;
  const valueFor = (source: string, target: string) => numberValue(chart.data.find((row) => row.source === source && row.target === target)?.value);
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="chart-svg chart-heatmap-svg" role="img" aria-label={chart.title}>
      {labels.map((label, index) => (
        <React.Fragment key={label}>
          <text x={pad + index * cell + cell / 2} y={pad - 12} textAnchor="middle" fontSize="10" fontWeight="600" fill="#526174">{shortLabel(label)}</text>
          <text x={pad - 10} y={pad + index * cell + cell / 2 + 4} textAnchor="end" fontSize="10" fontWeight="600" fill="#526174">{shortLabel(label)}</text>
        </React.Fragment>
      ))}
      {labels.flatMap((source, yIndex) =>
        labels.map((target, xIndex) => {
          const value = valueFor(source, target);
          const intensity = Math.min(Math.abs(value), 1);
          return (
            <rect
              key={`${source}-${target}`}
              x={pad + xIndex * cell}
              y={pad + yIndex * cell}
              width={cell - 4}
              height={cell - 4}
              rx={6}
              fill={value >= 0 ? `rgba(13, 148, 136, ${0.1 + intensity * 0.82})` : `rgba(225, 29, 72, ${0.1 + intensity * 0.82})`}
              stroke="rgba(255,255,255,0.9)"
            >
              <title>{`${source} × ${target}: ${formatAxisValue(value)}`}</title>
            </rect>
          );
        }),
      )}
    </svg>
  );
}

function structuredReportFromMetadata(metadata: Record<string, unknown>): StructuredReport | null {
  return structuredReportFromUnknown(metadata.structured_report);
}

function structuredReportFromUnknown(value: unknown): StructuredReport | null {
  if (!value || typeof value !== "object") return null;
  return value as StructuredReport;
}

function executionIssues(
  executionResult: Record<string, unknown> | undefined,
  key: string,
): { severity?: string; finding_ref?: string; issue: string; suggestion?: string }[] {
  const value = executionResult?.[key];
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      severity: typeof item.severity === "string" ? item.severity : undefined,
      finding_ref: typeof item.finding_ref === "string" ? item.finding_ref : undefined,
      issue: typeof item.issue === "string" ? item.issue : "",
      suggestion: typeof item.suggestion === "string" ? item.suggestion : undefined,
    }))
    .filter((item) => item.issue);
}

function arrayOfRecords(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object");
}

function metricValue(value: unknown): string {
  return typeof value === "number" ? String(value) : "-";
}

function axisTicks(max: number) {
  const upper = Math.max(max, 1);
  return [0, 0.25, 0.5, 0.75, 1].map((ratio) => upper * ratio);
}

function formatAxisValue(value: number) {
  if (Math.abs(value) >= 1000) return value.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(value < 10 ? 1 : 0);
}

function chartTypeLabel(chartType: Chart["chart_type"]) {
  const labels: Record<string, string> = {
    bar: "柱状图",
    line: "趋势图",
    pie: "占比图",
    histogram: "分布图",
    box_plot: "箱线图",
    correlation_heatmap: "相关性热力图",
  };
  return labels[chartType] ?? chartType;
}

function textAnalysisFromUnknown(value: unknown): TextAnalysisResult[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      task: typeof item.task === "string" ? item.task : "text_analysis",
      text_column: typeof item.text_column === "string" ? item.text_column : "text",
      group_column: typeof item.group_column === "string" ? item.group_column : null,
      summary: Boolean(item.summary) && typeof item.summary === "object" ? item.summary as Record<string, unknown> : {},
      insights: Array.isArray(item.insights) ? item.insights.map(String) : [],
      charts: Array.isArray(item.charts) ? item.charts as Chart[] : [],
    }));
}

function multimodalInputsFromMetadata(metadata: Record<string, unknown>): MultimodalInput[] {
  const value = metadata.multimodal_inputs;
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      kind: isMultimodalKind(item.kind) ? item.kind : "note",
      title: typeof item.title === "string" ? item.title : "未命名上下文",
      description: typeof item.description === "string" ? item.description : "",
      source_ref: typeof item.source_ref === "string" ? item.source_ref : null,
      media_type: typeof item.media_type === "string" ? item.media_type : null,
      data_url: typeof item.data_url === "string" ? item.data_url : null,
      processing_status: typeof item.processing_status === "string" ? item.processing_status : null,
      text_excerpt: typeof item.text_excerpt === "string" ? item.text_excerpt : null,
    }))
    .filter((item) => item.description);
}

function isMultimodalKind(value: unknown): value is MultimodalInput["kind"] {
  return value === "image" || value === "chart" || value === "pdf_page" || value === "screenshot" || value === "note";
}

function htmlReportForDownload(title: string, report: StructuredReport) {
  const findings = (report.key_findings ?? [])
    .map((finding) => `<section class="card"><h2>${escapeHtml(finding.title)}</h2><p>${escapeHtml(finding.content)}</p>${finding.evidence ? `<p class="muted">证据：${escapeHtml(finding.evidence)}</p>` : ""}${finding.recommended_action ? `<p class="muted">建议：${escapeHtml(finding.recommended_action)}</p>` : ""}</section>`)
    .join("");
  const charts = (report.charts ?? [])
    .map((chart) => {
      const displayScopeMessage = chartDisplayScopeMessage(chart);
      return `<section class="card"><h2>${escapeHtml(chart.title)}</h2><p class="muted">${escapeHtml(chart.chart_type)}</p>${chartSvgMarkup(chart)}${displayScopeMessage ? `<p>${escapeHtml(displayScopeMessage)}</p>` : ""}${chart.explanation ? `<p>${escapeHtml(chart.explanation)}</p>` : ""}</section>`;
    })
    .join("");
  const sql = rowsTableMarkup(report.sql_results ?? []);
  const gaps = (report.data_gaps ?? []).map((gap) => `<li>${escapeHtml(gap)}</li>`).join("");
  const issues = (report.validation_issues ?? []).map((issue) => `<li><b>${escapeHtml(issue.severity)}</b> · ${escapeHtml(issue.finding_ref)}: ${escapeHtml(issue.issue)}</li>`).join("");
  const trace = (report.analysis_trace ?? []).map((round) => {
    const summary = [
      round.plan?.route ? `route: ${round.plan.route}` : "",
      round.plan?.metric_column ? `metric: ${round.plan.metric_column}` : "",
      round.plan?.category_column ? `dimension: ${round.plan.category_column}` : "",
      typeof round.execution_result?.sql_row_count === "number" ? `sql rows: ${round.execution_result.sql_row_count}` : "",
      typeof round.execution_result?.chart_count === "number" ? `charts: ${round.execution_result.chart_count}` : "",
      round.execution_result?.python_source ? `python: ${String(round.execution_result.python_source)}` : "",
      round.execution_result?.fanout_mode ? `mode: ${String(round.execution_result.fanout_mode)}` : "",
    ].filter(Boolean).join(" · ");
    return `<section class="card"><h2>Round ${round.round_number}</h2>${summary ? `<p class="muted">${escapeHtml(summary)}</p>` : ""}<p>${escapeHtml(round.hypothesis?.statement ?? "")}</p><p class="muted">${escapeHtml(round.reflection?.insight_text ?? "")}</p></section>`;
  }).join("");
  const nextSteps = (report.recommended_next_steps ?? []).map((step) => `<li>${escapeHtml(step)}</li>`).join("");
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <style>
    body{margin:0;background:#f3f6fb;color:#111827;font-family:Arial,"Microsoft YaHei",sans-serif}
    main{max-width:1080px;margin:0 auto;padding:40px 28px 64px}
    h1{font-size:34px;margin:0 0 20px}.card{background:#fff;border:1px solid #dbe5f0;border-radius:8px;padding:20px;margin:14px 0}
    .summary{padding:24px 28px;background:#cfe6fb;border-radius:8px;font-size:17px;line-height:1.75}
    .muted{color:#64748b}.chart-grid{display:grid;grid-template-columns:280px 1fr;gap:20px;align-items:center}
    .legend{display:grid;gap:8px;font-size:14px}table{width:100%;border-collapse:collapse;background:white}
    th,td{padding:10px 12px;border-bottom:1px solid #e5edf5;text-align:left;font-size:14px}th{background:#111827;color:white}
  </style>
</head>
<body>
  <main>
    <h1>${escapeHtml(title)}</h1>
    <section class="summary">${escapeHtml(report.executive_summary)}</section>
    ${report.analysis_context ? `<h2>Analysis Context</h2><section class="card">${escapeHtml(report.analysis_context)}</section>` : ""}
    <h2>Key Findings</h2>${findings || "<p class='muted'>暂无核心发现。</p>"}
    <h2>Visualizations</h2>${charts || "<p class='muted'>暂无图表。</p>"}
    <h2>SQL Results</h2>${sql || "<p class='muted'>暂无 SQL 结果。</p>"}
    <h2>Data Gaps</h2>${gaps ? `<ul>${gaps}</ul>` : "<p class='muted'>暂无数据缺口。</p>"}
    <h2>Validation Issues</h2>${issues ? `<ul>${issues}</ul>` : "<p class='muted'>暂无校验问题。</p>"}
    <h2>Recommended Next Steps</h2>${nextSteps ? `<ul>${nextSteps}</ul>` : "<p class='muted'>暂无建议。</p>"}
    <h2>Analysis Trace</h2>${trace || "<p class='muted'>暂无分析轨迹。</p>"}
  </main>
</body>
</html>`;
}

function chartSvgMarkup(chart: Chart) {
  if (!chart.data.length) return "<p class='muted'>暂无图表数据。</p>";
  if (chart.chart_type === "pie") return pieSvgMarkup(chart);
  if (chart.chart_type === "line") return cartesianSvgMarkup(chart, "line");
  if (chart.chart_type === "histogram") return histogramSvgMarkup(chart);
  if (chart.chart_type === "box_plot") return boxSvgMarkup(chart);
  if (chart.chart_type === "correlation_heatmap") return heatmapSvgMarkup(chart);
  return cartesianSvgMarkup(chart, "bar");
}

function cartesianSvgMarkup(chart: Chart, mode: "bar" | "line") {
  const keys = Object.keys(chart.data[0] ?? {});
  const xKey = String(chart.spec.x ?? keys[0] ?? "x");
  const yKey = String(chart.spec.y ?? keys[keys.length - 1] ?? "y");
  const visibleRows = mode === "bar" && chart.data.length > 24
    ? [...chart.data].sort((leftRow, rightRow) => numberValue(rightRow[yKey]) - numberValue(leftRow[yKey]))
    : chart.data;
  const points = visibleRows
    .slice(0, 24)
    .map((row) => ({ label: String(row[xKey] ?? ""), value: numberValue(row[yKey]) }))
    .filter((point) => Number.isFinite(point.value));
  if (!points.length) return rowsTableMarkup(chart.data);
  const width = 720;
  const height = 280;
  const pad = 58;
  const max = Math.max(...points.map((point) => point.value), 1);
  const innerHeight = height - pad * 2;
  const step = (width - pad * 2) / Math.max(points.length, 1);
  let body = axisTicks(max)
    .map((tick) => {
      const y = height - pad - (tick / max) * innerHeight;
      return `<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="#e2e8f0"/><text x="${pad - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${escapeHtml(formatAxisValue(tick))}</text>`;
    })
    .join("");
  body += `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#cbd5e1"/>`;
  if (mode === "line") {
    const polyline = points
      .map((point, index) => `${pad + step * index + step / 2},${height - pad - (point.value / max) * innerHeight}`)
      .join(" ");
    body += `<polyline points="${polyline}" fill="none" stroke="${CHART_LINE_COLOR}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>`;
  } else {
    body += points
      .map((point, index) => {
        const barWidth = Math.max(step * 0.58, 6);
        const barHeight = (point.value / max) * innerHeight;
        const x = pad + step * index + (step - barWidth) / 2;
        const y = height - pad - barHeight;
        const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
        return `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="5" fill="${color}" fill-opacity="0.94"/>`;
      })
      .join("");
  }
  body += points
    .map((point, index) => `<text x="${pad + step * index + step / 2}" y="${height - 14}" text-anchor="middle" font-size="11" fill="#475569">${escapeHtml(shortLabel(point.label))}</text>`)
    .join("");
  return svgMarkup(width, height, body);
}

function chartDisplayScopeMessage(chart: Chart) {
  if (chart.chart_type !== "bar" || chart.data.length <= 24) return "";
  const keys = Object.keys(chart.data[0] ?? {});
  const yKey = String(chart.spec.y ?? keys[keys.length - 1] ?? "指标");
  return `图中按 ${yKey} 从高到低展示前 24 / ${chart.data.length} 个类别；完整结果见“查看图表数据”。`;
}

function pieSvgMarkup(chart: Chart) {
  const keys = Object.keys(chart.data[0] ?? {});
  const nameKey = String(chart.spec.names ?? keys[0] ?? "name");
  const valueKey = String(chart.spec.values ?? keys[keys.length - 1] ?? "value");
  const slices = chart.data
    .slice(0, 8)
    .map((row) => ({ label: String(row[nameKey] ?? ""), value: Math.max(numberValue(row[valueKey]), 0) }))
    .filter((slice) => slice.value > 0);
  const total = slices.reduce((sum, slice) => sum + slice.value, 0);
  if (!total) return rowsTableMarkup(chart.data);
  let current = 0;
  const paths = slices
    .map((slice, index) => {
      const start = current;
      const end = current + (slice.value / total) * Math.PI * 2;
      current = end;
      return `<path d="${arcPath(110, 110, 82, start, end)}" fill="${CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]}" stroke="white" stroke-width="3"/>`;
    })
    .join("");
  const legend = slices
    .map((slice, index) => `<div><span style="display:inline-block;width:11px;height:11px;background:${CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]};margin-right:8px"></span>${escapeHtml(slice.label)} <b>${((slice.value / total) * 100).toFixed(1)}%</b></div>`)
    .join("");
  return `<div class="chart-grid">${svgMarkup(220, 220, `${paths}<circle cx="110" cy="110" r="44" fill="white"/>`)}<div class="legend">${legend}</div></div>`;
}

function histogramSvgMarkup(chart: Chart) {
  if (chart.spec.y) return cartesianSvgMarkup(chart, "bar");
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "value");
  const values = chart.data.map((row) => numberValue(row[xKey])).filter(Number.isFinite);
  if (!values.length) return rowsTableMarkup(chart.data);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const buckets = Array.from({ length: 10 }, (_, index) => ({ label: String(index + 1), value: 0 }));
  values.forEach((value) => {
    const ratio = max === min ? 0 : (value - min) / (max - min);
    buckets[Math.min(9, Math.floor(ratio * 10))].value += 1;
  });
  return cartesianSvgMarkup({ ...chart, spec: { x: "label", y: "value" }, data: buckets }, "bar");
}

function boxSvgMarkup(chart: Chart) {
  const keys = Object.keys(chart.data[0] ?? {});
  const xKey = String(chart.spec.x ?? keys[0] ?? "category");
  const yKey = String(chart.spec.y ?? keys[keys.length - 1] ?? "value");
  const groups = new Map<string, number[]>();
  chart.data.forEach((row) => {
    const value = numberValue(row[yKey]);
    if (Number.isFinite(value)) groups.set(String(row[xKey] ?? "未分组"), [...(groups.get(String(row[xKey] ?? "未分组")) ?? []), value]);
  });
  const summaries = Array.from(groups.entries()).slice(0, 8).map(([label, values]) => ({ label, ...quartiles(values) }));
  if (!summaries.length) return rowsTableMarkup(chart.data);
  const width = 720;
  const height = 280;
  const pad = 58;
  const max = Math.max(...summaries.map((item) => item.max), 1);
  const step = (width - pad * 2) / summaries.length;
  const sy = (value: number) => height - pad - (value / max) * (height - pad * 2);
  let body = axisTicks(max)
    .map((tick) => {
      const y = sy(tick);
      return `<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="#e2e8f0"/><text x="${pad - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${escapeHtml(formatAxisValue(tick))}</text>`;
    })
    .join("");
  body += `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#cbd5e1"/>`;
  body += summaries
    .map((item, index) => {
      const x = pad + step * index + step / 2;
      const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
      return `<line x1="${x}" x2="${x}" y1="${sy(item.min)}" y2="${sy(item.max)}" stroke="${color}" stroke-width="3"/><rect x="${x - 20}" y="${sy(item.q3)}" width="40" height="${Math.max(sy(item.q1) - sy(item.q3), 2)}" fill="${color}" fill-opacity="0.28" stroke="${color}"/><line x1="${x - 24}" x2="${x + 24}" y1="${sy(item.median)}" y2="${sy(item.median)}" stroke="#111827" stroke-width="3"/><text x="${x}" y="${height - 14}" text-anchor="middle" font-size="11" fill="#475569">${escapeHtml(shortLabel(item.label))}</text>`;
    })
    .join("");
  return svgMarkup(width, height, body);
}

function heatmapSvgMarkup(chart: Chart) {
  const labels = Array.from(new Set(chart.data.flatMap((row) => [String(row.source ?? ""), String(row.target ?? "")]))).filter(Boolean).slice(0, 10);
  if (!labels.length) return rowsTableMarkup(chart.data);
  const cell = 44;
  const pad = 90;
  const size = pad + labels.length * cell + 16;
  const valueFor = (source: string, target: string) => numberValue(chart.data.find((row) => row.source === source && row.target === target)?.value);
  const body = labels
    .flatMap((source, yIndex) =>
      labels.map((target, xIndex) => {
        const value = valueFor(source, target);
        const intensity = Math.min(Math.abs(value), 1);
        const color = value >= 0 ? `rgba(15,118,110,${0.12 + intensity * 0.78})` : `rgba(220,38,38,${0.12 + intensity * 0.78})`;
        return `<rect x="${pad + xIndex * cell}" y="${pad + yIndex * cell}" width="${cell - 2}" height="${cell - 2}" fill="${color}"/>`;
      }),
    )
    .join("");
  return svgMarkup(size, size, body);
}

function rowsTableMarkup(rows: Record<string, unknown>[]) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0] ?? {}).slice(0, 8);
  return `<table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${rows
    .slice(0, 60)
    .map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function svgMarkup(width: number, height: number, body: string) {
  return `<svg viewBox="0 0 ${width} ${height}" style="width:100%;max-height:360px;background:#f8fafc;border-radius:8px;margin:12px 0">${body}</svg>`;
}

function escapeHtml(value: string) {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function numberValue(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "string") return Number(value.replace(/,/g, ""));
  return Number.NaN;
}

function shortLabel(value: string) {
  return value.length > 10 ? `${value.slice(0, 9)}…` : value;
}

function valueText(value: number | string | null | undefined) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
}

function arcPath(cx: number, cy: number, radius: number, start: number, end: number) {
  const startX = cx + radius * Math.cos(start);
  const startY = cy + radius * Math.sin(start);
  const endX = cx + radius * Math.cos(end);
  const endY = cy + radius * Math.sin(end);
  const largeArc = end - start > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} 1 ${endX} ${endY} Z`;
}

function quartiles(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    min: sorted[0] ?? 0,
    q1: percentile(sorted, 0.25),
    median: percentile(sorted, 0.5),
    q3: percentile(sorted, 0.75),
    max: sorted[sorted.length - 1] ?? 0,
  };
}

function percentile(sorted: number[], p: number) {
  if (!sorted.length) return 0;
  const index = (sorted.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? sorted[lower] ?? 0) * weight;
}

function DownloadButton({ label, fileName, content, mime }: { label: string; fileName: string; content: string; mime: string }) {
  const href = URL.createObjectURL(new Blob([content], { type: `${mime};charset=utf-8` }));
  return (
    <a href={href} download={fileName} className="small-button inline-flex">
      {label}
    </a>
  );
}

function Alert({ children, tone = "info" }: { children: React.ReactNode; tone?: "info" | "error" }) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      aria-live={tone === "error" ? "assertive" : "polite"}
      className={`mt-4 rounded-xl border px-4 py-3 text-sm ${tone === "error" ? "border-rose-200 bg-rose-50 text-rose-950" : "border-sky-200 bg-sky-50 text-slate-900"}`}
    >
      {children}
    </div>
  );
}

function LoadingLine() {
  return <div className="rounded-xl border border-line bg-white px-4 py-3 text-sm font-bold text-slate-500 shadow-[0_8px_24px_rgba(15,23,42,0.04)]">正在从数据库加载...</div>;
}

async function pollAnalysisJob(
  jobId: string,
  onUpdate: (job: AnalysisJob) => void,
): Promise<AnalysisJob> {
  const initial = await apiGet<AnalysisJob>(`/analysis/jobs/${jobId}`);
  onUpdate(initial);
  if (!isActiveAnalysisJob(initial)) return initial;
  if (typeof EventSource !== "undefined") {
    try {
      return await streamAnalysisJob(jobId, initial, onUpdate);
    } catch (error) {
      console.warn("Workflow event stream unavailable; falling back to polling.", error);
    }
  }
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const job = await apiGet<AnalysisJob>(`/analysis/jobs/${jobId}`);
    onUpdate(job);
    if (!isActiveAnalysisJob(job)) return job;
    await delay(1000);
  }
  throw new Error("分析任务仍在运行，请稍后在任务列表中查看结果。");
}

function streamAnalysisJob(
  jobId: string,
  initial: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
): Promise<AnalysisJob> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let refreshing = false;
    let terminalEventReceived = false;
    let pollTimer: number | null = null;
    let lastJob = initial;
    const afterSequence = initial.last_event_sequence ?? 0;
    const streamUrl = `${API_BASE_URL}/analysis/jobs/${jobId}/events?after_sequence=${afterSequence}`;
    const source = new EventSource(streamUrl, { withCredentials: true });
    const timeout = window.setTimeout(() => finish(new Error("Workflow event stream timed out.")), 600000);

    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      source.close();
      window.clearTimeout(timeout);
      if (pollTimer !== null) window.clearInterval(pollTimer);
      if (error) reject(error);
      else resolve(lastJob);
    };

    const refreshJob = async () => {
      if (refreshing || settled) return;
      refreshing = true;
      try {
        lastJob = await apiGet<AnalysisJob>(`/analysis/jobs/${jobId}`);
        onUpdate(lastJob);
        if (!isActiveAnalysisJob(lastJob)) finish();
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      } finally {
        refreshing = false;
      }
    };

    source.addEventListener("workflow", () => void refreshJob());
    source.addEventListener("end", () => {
      terminalEventReceived = true;
      source.close();
      void refreshJob();
    });
    source.onerror = () => {
      if (terminalEventReceived) return;
      finish(new Error("Workflow event stream disconnected."));
    };
    pollTimer = window.setInterval(() => void refreshJob(), 2000);
  });
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function multimodalKindLabel(kind: MultimodalInput["kind"]) {
  const labels: Record<MultimodalInput["kind"], string> = {
    image: "图片",
    chart: "图表",
    pdf_page: "PDF/文档页",
    screenshot: "截图",
    note: "文本备注",
  };
  return labels[kind] ?? kind;
}

function multimodalStatusLabel(item: MultimodalInput) {
  const status = item.processing_status;
  if (status === "native_image_payload") return "已进入 Kimi 视觉上下文";
  if (status === "pdf_text_extracted") return "PDF 文本已抽取";
  if (status === "pdf_text_unavailable") return "PDF 文本未抽取";
  if (status === "text_context") return "已作为文本上下文";
  if (item.data_url && item.media_type?.startsWith("image/")) return "已进入 Kimi 视觉上下文";
  if (item.kind === "pdf_page") return "PDF 待抽取/已降级";
  return "已作为上下文";
}

function multimodalStatusClass(status?: string | null) {
  if (status === "native_image_payload" || status === "pdf_text_extracted") {
    return "bg-emerald-50 text-emerald-700";
  }
  if (status === "pdf_text_unavailable") {
    return "bg-amber-50 text-amber-700";
  }
  return "bg-blue-50 text-blue-700";
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("文件读取失败。"));
    reader.readAsDataURL(file);
  });
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

function formatTime(value?: string | null): string {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value.replace("T", " ").slice(0, 19);
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  })
    .format(parsed)
    .replace(/\//g, "-");
}

function latestReportCaption(reports: Report[]) {
  if (!reports.length) return "等待生成";
  return `最新: ${formatTime(reports[0].created_at)}`;
}

function translateStatus(value: string) {
  return { imported: "已导入", profiled: "已画像", cleaned: "已清洗", failed: "失败" }[value] ?? value;
}

function uploadStatusLabel(value: UploadQueueItem["status"]) {
  return {
    ready: "等待处理",
    previewing: "读取 Sheet",
    uploading: "导入中",
    cleaning: "清洗中",
    done: "完成",
    error: "失败",
  }[value];
}

function relationshipKey(
  relationship: Pick<DatasetRelationshipPlan, "left_dataset_id" | "right_dataset_id" | "left_column" | "right_column">,
) {
  return `${relationship.left_dataset_id}:${relationship.left_column}->${relationship.right_dataset_id}:${relationship.right_column}`;
}

function formatRelationship(
  relationship: Pick<DatasetRelationshipPlan, "left_dataset_id" | "right_dataset_id" | "left_column" | "right_column" | "join_type">,
  tables: DatasetGroupTable[] = [],
) {
  const names = new Map(
    tables.map((table) => [table.dataset.dataset_id, compactDatasetName(table.dataset.name)]),
  );
  const left = names.get(relationship.left_dataset_id);
  const right = names.get(relationship.right_dataset_id);
  return `${left ? `${left}.` : ""}${relationship.left_column} -> ${right ? `${right}.` : ""}${relationship.right_column} (${relationship.join_type})`;
}

function compactDatasetName(name: string) {
  return name.replace(/\.(csv|xlsx|json|txt)$/i, "").replace(/_dataset$/i, "");
}

function relationshipMatchRate(
  relationship: Pick<DatasetRelationshipPlan, "last_match_rate" | "baseline_match_rate">,
) {
  return relationship.last_match_rate ?? relationship.baseline_match_rate ?? null;
}

function formatRelationshipMetrics(
  relationship: Pick<DatasetRelationshipPlan, "confidence" | "last_match_rate" | "baseline_match_rate">,
) {
  const confidence = relationship.confidence;
  const matchRate = relationshipMatchRate(relationship);
  const metrics = [
    confidence == null ? null : `推荐置信度 ${(confidence * 100).toFixed(0)}%`,
    matchRate == null ? null : `样本匹配率 ${(matchRate * 100).toFixed(0)}%`,
  ].filter(Boolean);
  return metrics.length ? metrics.join(" · ") : "已自动确认";
}

function errorMessage(error: unknown) {
  if (error instanceof DOMException && error.name === "AbortError") return "请求超时，请稍后重试。";
  if (error instanceof Error) return error.message;
  return String(error);
}

function dashboardSyncErrorMessage(error: string) {
  if (/type error|failed to fetch|load failed|network|无法连接后端|数据同步暂时中断/i.test(error)) {
    return "网络连接出现短暂波动，已有数据不会受影响。";
  }
  return error;
}

function loginErrorMessage(error: unknown) {
  const message = errorMessage(error);
  if (/401|invalid username or password/i.test(message)) {
    return "用户名或密码不正确，请检查后重试。";
  }
  if (/429|rate limit|too many requests/i.test(message)) {
    return "登录尝试过于频繁，请稍后再试。";
  }
  if (/422/.test(message)) {
    return "用户名或密码格式不符合要求，请检查后重试。";
  }
  return message;
}

async function analysisErrorMessage(error: unknown) {
  const message = errorMessage(error);
  if (!message.startsWith("无法连接后端服务")) return message;
  try {
    await apiGet("/health");
    return "后端健康检查通过，但分析请求连接中断。请稍后再运行一次；如果持续出现，查看 data/runtime/backend.err.log 中的后端异常。";
  } catch {
    return message;
  }
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
