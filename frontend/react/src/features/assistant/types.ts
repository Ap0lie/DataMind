export type AssistantScopeType = "auto" | "dataset" | "dataset_group" | "report";
export type AssistantExecutionMode = "ask" | "execute";
export type AssistantCapability = "data_prepare" | "relationship_manage" | "analysis_manage" | "report_manage" | "semantic_manage" | "asset_recycle";
export type AssistantAssetType = "dataset" | "dataset_group" | "report" | "semantic_model";

export const ASSISTANT_FULL_CAPABILITIES: AssistantCapability[] = [
  "data_prepare",
  "relationship_manage",
  "analysis_manage",
  "report_manage",
  "semantic_manage",
  "asset_recycle",
];

export type AssistantConversation = {
  conversation_id: string;
  title: string;
  scope_type: AssistantScopeType;
  scope_id?: string | null;
  summary: string;
  active_run_id?: string | null;
  created_at: string;
  updated_at: string;
  last_message_at?: string | null;
};

export type AssistantCitation = {
  source_type: "dataset" | "analysis_job" | "report";
  source_id: string;
  label: string;
  excerpt: string;
  dataset_id?: string | null;
  artifact_role?: "evidence" | "deliverable";
};

export type AssistantAttachment = {
  attachment_id: string;
  conversation_id: string;
  message_id?: string | null;
  file_name: string;
  media_type: string;
  size_bytes: number;
  width: number;
  height: number;
  attachment_kind: "image" | "data_file";
  import_status?: string | null;
  dataset_id?: string | null;
  import_batch_id?: string | null;
  created_at: string;
  content_url: string;
};

export type AssistantMessage = {
  message_id: string;
  conversation_id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  status: string;
  provider?: string | null;
  model?: string | null;
  citations: AssistantCitation[];
  attachments: AssistantAttachment[];
  metadata: Record<string, unknown>;
  created_at: string;
};

export type AssistantRun = {
  run_id: string;
  conversation_id: string;
  user_message_id: string;
  assistant_message_id: string;
  status: string;
  current_stage: string;
  analysis_job_id?: string | null;
  pending_confirmation: Record<string, unknown>;
  execution_mode: AssistantExecutionMode;
  execution_plan: Record<string, unknown>;
  current_action_id?: string | null;
  required_permission?: AssistantCapability | null;
  error?: string | null;
  last_event_sequence: number;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
};

export type AssistantEvent = {
  sequence: number;
  event_type: string;
  status: string;
  message: string;
  tool_name?: string | null;
  payload: Record<string, unknown>;
  created_at: string;
};

export type AssistantScopeAsset = { id: string; name: string };

export type AssistantPermissionGrant = {
  grant_id: string;
  asset_type: AssistantAssetType;
  asset_id: string;
  capabilities: AssistantCapability[];
  status: string;
  created_at: string;
  revoked_at?: string | null;
};

export type AssistantAction = {
  action_id: string;
  run_id?: string | null;
  conversation_id?: string | null;
  tool_name: string;
  status: string;
  asset_type?: AssistantAssetType | null;
  asset_id?: string | null;
  reversible: boolean;
  undone_at?: string | null;
  result: Record<string, unknown>;
  error?: string | null;
  created_at: string;
  completed_at?: string | null;
};

export type RecycledAsset = {
  asset_type: AssistantAssetType;
  asset_id: string;
  name: string;
  deleted_at: string;
  purge_after: string;
};

export type AssistantImportFilePreview = {
  attachment_id: string;
  file_name: string;
  source_type: string;
  valid: boolean;
  error?: string | null;
  row_count: number;
  column_count: number;
  columns: string[];
  preview_records: Record<string, unknown>[];
  selected_sheet?: string | null;
  requires_sheet_selection?: boolean;
  sheets?: Array<{ sheet_name: string; row_count: number; column_count: number; selected?: boolean }>;
};

export type AssistantImportBatch = {
  batch_id: string;
  conversation_id: string;
  attachment_ids: string[];
  status: string;
  preview: { files?: AssistantImportFilePreview[]; valid_count?: number; invalid_count?: number };
  dataset_ids: string[];
  dataset_group_id?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
};
