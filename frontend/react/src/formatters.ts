import type {
  DatasetGroupTable,
  DatasetRelationshipPlan,
  Report,
  UploadQueueItem,
} from "./domain-types";

export function formatTime(value?: string | null): string {
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

export function latestReportCaption(reports: Report[]) {
  if (!reports.length) return "等待生成";
  return `最新: ${formatTime(reports[0].created_at)}`;
}

export function translateStatus(value: string) {
  return { imported: "已导入", profiled: "已画像", cleaned: "已清洗", failed: "失败" }[value] ?? value;
}

export function uploadStatusLabel(value: UploadQueueItem["status"]) {
  return {
    ready: "等待处理",
    previewing: "读取 Sheet",
    uploading: "导入中",
    cleaning: "清洗中",
    done: "完成",
    error: "失败",
  }[value];
}

export function cleaningStageLabel(stage: string) {
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

export function assistantStageLabel(stage: string) {
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

export function relationshipKey(
  relationship: Pick<
    DatasetRelationshipPlan,
    "left_dataset_id" | "right_dataset_id" | "left_column" | "right_column"
  >,
) {
  return `${relationship.left_dataset_id}:${relationship.left_column}->${relationship.right_dataset_id}:${relationship.right_column}`;
}

export function formatRelationship(
  relationship: Pick<
    DatasetRelationshipPlan,
    "left_dataset_id" | "right_dataset_id" | "left_column" | "right_column" | "join_type"
  >,
  tables: DatasetGroupTable[] = [],
) {
  const names = new Map(
    tables.map((table) => [table.dataset.dataset_id, compactDatasetName(table.dataset.name)]),
  );
  const left = names.get(relationship.left_dataset_id);
  const right = names.get(relationship.right_dataset_id);
  return `${left ? `${left}.` : ""}${relationship.left_column} -> ${right ? `${right}.` : ""}${relationship.right_column} (${relationship.join_type})`;
}

export function compactDatasetName(name: string) {
  return name.replace(/\.(csv|xlsx|json|txt)$/i, "").replace(/_dataset$/i, "");
}

export function relationshipMatchRate(
  relationship: Pick<DatasetRelationshipPlan, "last_match_rate" | "baseline_match_rate">,
) {
  return relationship.last_match_rate ?? relationship.baseline_match_rate ?? null;
}

export function formatRelationshipMetrics(
  relationship: Pick<
    DatasetRelationshipPlan,
    "confidence" | "last_match_rate" | "baseline_match_rate"
  >,
) {
  const confidence = relationship.confidence;
  const matchRate = relationshipMatchRate(relationship);
  const metrics = [
    confidence == null ? null : `推荐置信度 ${(confidence * 100).toFixed(0)}%`,
    matchRate == null ? null : `样本匹配率 ${(matchRate * 100).toFixed(0)}%`,
  ].filter(Boolean);
  return metrics.length ? metrics.join(" · ") : "已自动确认";
}

export function errorMessage(error: unknown) {
  if (error instanceof DOMException && error.name === "AbortError") return "请求超时，请稍后重试。";
  if (error instanceof Error) return error.message;
  return String(error);
}

export function dashboardSyncErrorMessage(error: string) {
  if (/type error|failed to fetch|load failed|network|无法连接后端|数据同步暂时中断/i.test(error)) {
    return "网络连接出现短暂波动，已有数据不会受影响。";
  }
  return error;
}

export function loginErrorMessage(error: unknown) {
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
