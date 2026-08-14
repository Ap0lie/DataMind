import type {
  AnalysisContract,
  AnalysisLineage,
  StatisticalVerification,
} from "./features/analysis/AnalysisReliabilityPanel";
import type { CleaningRunDetail } from "./features/datasets/CleaningWorkspace";

export type Page = "首页" | "数据集" | "分析任务" | "报告" | "Kimi";
export type DatasetWorkspaceView = "assets" | "relationships" | "detail";

export type Dataset = {
  dataset_id: string;
  user_id?: string;
  name: string;
  source_type: string;
  status: string;
  source_metadata: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

export type DatasetColumnProfile = {
  name: string;
  dtype: string;
  missing_count: number;
  distinct_count: number;
  is_numeric: boolean;
  min_value?: number | null;
  max_value?: number | null;
  mean?: number | null;
};

export type DatasetProfile = {
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

export type DatasetJoinConfig = {
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

export type DatasetRelationshipPlan = DatasetJoinConfig & {
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

export type DatasetReference = {
  dataset_id: string;
  name: string;
  status: string;
  row_count: number;
  column_count: number;
  columns: string[];
};

export type JoinSuggestionCandidate = DatasetJoinConfig & {
  score: number;
  reason: string;
  left_type: string;
  right_type: string;
  left_role: string;
  right_role: string;
  estimated_match_rate: number;
};

export type DatasetRelationshipCandidate = DatasetRelationshipPlan & {
  confidence: number;
  source: "rules" | "llm" | "validated_llm";
  estimated_match_rate: number;
  left_type?: string;
  right_type?: string;
  left_role?: string;
  right_role?: string;
};

export type DatasetGroupTable = {
  dataset: Dataset;
  row_count: number;
  column_count: number;
  columns: string[];
  entity_type: "fact" | "dimension" | "bridge" | "lookup" | "wide" | "unknown";
  sample_records: Record<string, unknown>[];
};

export type DatasetGroup = {
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

export type DatasetRelationshipSuggestionResponse = {
  group: DatasetGroup;
  candidates: DatasetRelationshipCandidate[];
  llm_used: boolean;
  compact_context: Record<string, unknown>;
  validation_issues: string[];
};

export type DatasetRelationshipAutoConfigureResponse = DatasetRelationshipSuggestionResponse & {
  saved_relationships: DatasetRelationshipPlan[];
  primary_dataset_id: string | null;
  unresolved_dataset_ids: string[];
};

export type PlannerDecision = {
  decision_id: string;
  semantic_source: string;
  semantic_model_id?: string | null;
  semantic_model_version?: number | null;
  semantic_plan: Record<string, unknown>;
  confidence_breakdown: Record<string, number | null>;
  raw_confidence: number;
  calibrated_confidence: number;
  confidence_level: "low" | "medium" | "high";
  requires_confirmation: boolean;
  ambiguities: string[];
  evidence: string[];
};

export type JoinSuggestionResponse = {
  primary_dataset: DatasetReference;
  additional_datasets: DatasetReference[];
  suggestions: JoinSuggestionCandidate[];
  validation_issues: { severity: string; finding_ref: string; issue: string; suggestion?: string }[];
};

export type MultiDatasetContext = {
  primary_dataset: DatasetReference;
  additional_datasets: DatasetReference[];
  join_plan: DatasetJoinConfig[];
  join_summary: Record<string, unknown>;
  joined_profile?: DatasetProfile | null;
  column_source_map: Record<string, string>;
  validation_issues: { severity: string; finding_ref: string; issue: string; suggestion?: string }[];
};

export type DatasetDetail = {
  profile: DatasetProfile | null;
  rawRecords: Record<string, unknown>[];
  cleanedRecords: Record<string, unknown>[];
  analysisRecords: Record<string, unknown>[];
  cleaningRuns: CleaningRunDetail[];
  columnMetadata: DatasetColumnMetadata[];
};

export type Report = {
  id: string;
  dataset_id: string;
  title: string;
  markdown: string;
  metadata: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
  version?: number;
};

export type DatasetColumnMetadata = {
  column_name: string;
  inferred_type: string;
  override_type?: string | null;
  description: string;
  role: "dimension" | "metric" | "id" | "text" | "date" | "ignore";
  created_at?: string | null;
  updated_at?: string | null;
};

export type ReportVersionSummary = {
  report_id: string;
  dataset_id: string;
  title: string;
  question?: string | null;
  version: number;
  created_at?: string | null;
  updated_at?: string | null;
};

export type ExcelSheetPreview = {
  sheet_name: string;
  row_count: number;
  column_count: number;
  score: number;
  selected: boolean;
  preview_records: Record<string, unknown>[];
};

export type UploadQueueItem = {
  id: string;
  file: File;
  sheetPreviews: ExcelSheetPreview[];
  selectedSheetName: string;
  status: "ready" | "previewing" | "uploading" | "cleaning" | "done" | "error";
  message?: string;
  inserted?: number;
  datasetId?: string;
};

export type DatasetImportPipelineState = {
  phase: "idle" | "processing" | "creating_group" | "relationships" | "complete" | "attention" | "error";
  message: string;
  tableCount: number;
  relationshipCount: number;
  unresolvedCount: number;
  llmUsed: boolean;
};

export type Chart = {
  title: string;
  chart_type: string;
  spec: Record<string, unknown>;
  data: Record<string, unknown>[];
  explanation?: string;
  related_finding_ids?: string[];
};

export type TextAnalysisResult = {
  task: string;
  text_column: string;
  group_column?: string | null;
  summary: Record<string, unknown>;
  insights: string[];
  charts: Chart[];
};

export type PythonCodeAttempt = {
  attempt: number;
  status: "failed" | "succeeded";
  code?: string | null;
  error?: string | null;
  provider?: string | null;
  model?: string | null;
};

export type StructuredReport = {
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

export type AnalysisResponse = {
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

export type PlannerMetadata = {
  confidence: number;
  route_reason: string;
  candidate_metrics: string[];
  candidate_dimensions: string[];
  candidate_time_fields: string[];
  candidate_text_fields: string[];
  clarifying_questions: string[];
  multi_dataset_summary?: Record<string, unknown>;
};

export type WorkflowTraceNode = {
  node: string;
  status: string;
  provider?: string | null;
  model?: string | null;
  input_summary: string;
  output_summary: string;
  fallback?: string | null;
  error?: string | null;
};

export type AnalysisJobEvent = {
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

export type AnalysisJob = {
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

export type MultimodalInput = {
  kind: "image" | "chart" | "pdf_page" | "screenshot" | "note";
  title: string;
  description: string;
  source_ref?: string | null;
  media_type?: string | null;
  data_url?: string | null;
  processing_status?: string | null;
  text_excerpt?: string | null;
};

export type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

export type ActiveTask = {
  id: string;
  kind: "analysis" | "cleaning" | "assistant";
  page: Page;
  title: string;
  stage: string;
  progress: number;
  updatedAt: string;
};
