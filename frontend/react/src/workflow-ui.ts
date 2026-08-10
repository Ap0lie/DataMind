export const WORKFLOW_NODE_STATUS = {
  waiting: "waiting",
  running: "running",
  completed: "completed",
  failed: "failed",
} as const;

export const AGENT_WORKFLOW_STEPS = [
  { key: "planner", label: "规划器", logLabel: "规划器", stages: ["queued", "starting", "planner"], detail: "理解问题、检查输入并规划分析路线。" },
  { key: "structure", label: "结构理解", logLabel: "结构理解", stages: ["design_framework", "join_prepare"], detail: "解析字段、画像、数据包关系和分析框架。" },
  { key: "sql", label: "SQL 智能体", logLabel: "SQL 智能体", stages: ["sql_agent", "loop_bootstrap", "loop_decide", "loop_execute", "loop_observe", "loop_verify", "loop_repair", "loop_fallback"], detail: "生成安全 SELECT 并在内部 dataset 表上执行。" },
  {
    key: "python",
    label: "Python 智能体",
    logLabel: "Python 智能体",
    stages: ["python_agent", "iterative_prepare_rounds", "iterative_round_1", "iterative_fanout_round", "iterative_reflect_and_merge", "loop_finalize", "loop_adversarial_repair", "integrate_insights"],
    detail: "执行统计、探索分析、文本分析和多轮洞察整合。",
  },
  { key: "visualization", label: "可视化智能体", logLabel: "可视化智能体", stages: ["format_charts"], detail: "整理图表规格、标题和报告图表说明。" },
  { key: "reviewer", label: "审查器", logLabel: "审查器", stages: ["statistical_verify", "adversarial_validate"], detail: "检查统计支持、分析粒度、数据缺口和可追溯性风险。" },
  { key: "report", label: "报告", logLabel: "报告智能体", stages: ["report_agent", "report_decide", "report_execute", "report_verify", "report_repair", "report_fallback", "report_commit", "complete"], detail: "自主选择报告策略、验证证据并幂等提交。" },
] as const;

export const AGENT_PLAN_STEPS = [
  { key: "sql", label: "SQL 分析", workflowKeys: ["sql"] },
  { key: "explore", label: "探索分析", workflowKeys: ["python"] },
  { key: "visualize", label: "可视化", workflowKeys: ["visualization"] },
  { key: "report", label: "报告", workflowKeys: ["reviewer", "report"] },
] as const;

export type WorkflowNodeStatus = (typeof WORKFLOW_NODE_STATUS)[keyof typeof WORKFLOW_NODE_STATUS];
export type AgentWorkflowStepKey = (typeof AGENT_WORKFLOW_STEPS)[number]["key"];

export type WorkflowEvent = {
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

export type WorkflowJob = {
  status: string;
  current_stage: string;
  events: WorkflowEvent[];
  error?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  agent_mode?: "legacy" | "loop";
  loop_summary?: Record<string, unknown>;
  loop_terminal_reason?: string | null;
  report_strategy?: string | null;
  report_revision_count?: number;
  report_terminal_reason?: string | null;
};

export type AgentWorkflowStepView = Omit<(typeof AGENT_WORKFLOW_STEPS)[number], "label" | "detail"> & {
  label: string;
  detail: string;
  status: WorkflowNodeStatus;
  events: WorkflowEvent[];
};

export type WorkflowLogEntry = {
  icon: string;
  text: string;
  kind: "info" | "running" | "completed" | "failed";
  createdAt: string;
};

export type AnalysisComponents = {
  sql: boolean;
  python: boolean;
};

export const WORKFLOW_TOOL_LABELS: Record<string, string> = {
  inspect_analysis_context: "分析上下文检查",
  inspect_source_datasets: "源数据检查",
  profile_dataset: "数据画像检查",
  aggregate_dataset: "数据聚合",
  aggregate_source_dataset: "源表粒度聚合",
  detect_anomalies: "异常检测",
  analyze_text: "文本分析",
  execute_semantic_query: "语义查询",
  execute_safe_sql: "安全 SQL 分析",
  execute_python_analysis: "Python 分析",
  generate_chart: "图表生成",
  validate_evidence: "证据校验",
  legacy_fallback: "确定性规则分析",
};

export const LOOP_EVENT_LABELS: Record<string, string> = {
  loop_bootstrap: "准备",
  decision: "决策",
  deferred_decision: "顺序执行",
  normalized_decision: "顺序编排",
  invalid_decision: "重新决策",
  tool_execution: "工具调用",
  duplicate_action: "结果复用",
  observation: "观察",
  verification: "验证",
  repair: "自动修复",
  fallback: "规则降级",
  adversarial_repair: "审查返工",
  statistical_preflight: "统计预审",
  statistical_validation: "统计审查",
  loop_finalize: "完成",
  provider_error: "模型异常后规则执行",
  report_decision: "报告决策",
  report_draft: "生成草稿",
  report_validation: "报告验证",
  report_repair: "报告修订",
  evidence_request: "请求补证据",
  report_fallback: "模板降级",
  report_commit: "提交报告",
};

export function successfulLoopToolNames(events: WorkflowEvent[]) {
  return Array.from(new Set(
    events
      .filter((event) => (
        event.tool_name
        && ["succeeded", "completed"].includes(event.status)
        && ["observation", "tool_execution", "fallback"].includes(event.event_type ?? "")
      ))
      .map((event) => String(event.tool_name)),
  ));
}

export function loopAnalysisComponents(job: WorkflowJob): AnalysisComponents | null {
  const summaryComponents = job.loop_summary?.analysis_components;
  if (Array.isArray(summaryComponents)) {
    const normalized = summaryComponents.map((value) => String(value).toLowerCase());
    return { sql: normalized.includes("sql"), python: normalized.includes("python") };
  }

  const hasExecutionFact = job.events.some((event) =>
    ["observation", "tool_execution", "fallback"].includes(event.event_type ?? ""),
  );
  if (!hasExecutionFact) return null;
  const tools = successfulLoopToolNames(job.events);
  const fallback = tools.includes("legacy_fallback");
  return {
    sql: fallback || tools.includes("execute_safe_sql") || tools.includes("execute_semantic_query"),
    python: fallback || tools.includes("execute_python_analysis"),
  };
}

export function deriveAnalysisComponents(
  job: WorkflowJob,
  metadata: Record<string, unknown> = {},
): AnalysisComponents {
  const loopMode = job.agent_mode === "loop" || metadata.agent_mode === "loop";
  const loopComponents = loopMode ? loopAnalysisComponents(job) : null;
  if (loopComponents) return loopComponents;
  if (loopMode) return { sql: false, python: false };

  const route = String(metadata.route ?? "").toLowerCase();
  const nodes = Array.isArray(metadata.nodes)
    ? metadata.nodes.map((node) => String(node).toLowerCase())
    : [];
  const eventSignals = job.events.flatMap((event) => [event.node, event.stage, event.tool_name])
    .filter((value): value is string => Boolean(value))
    .map((value) => value.toLowerCase());
  const sourceWasUsed = (key: "sql_source" | "python_source") => {
    const value = String(metadata[key] ?? "").trim().toLowerCase();
    return Boolean(value && !["none", "disabled", "not_run"].includes(value));
  };
  const signals = [...nodes, ...eventSignals];
  const executed = {
    sql: sourceWasUsed("sql_source")
      || signals.some((signal) => signal === "sql_agent" || signal === "execute_safe_sql"),
    python: sourceWasUsed("python_source")
      || signals.some((signal) => signal === "python_agent" || signal === "execute_python_analysis"),
  };
  return {
    sql: route === "sql" || route === "hybrid" || executed.sql,
    python: route === "python" || route === "hybrid" || executed.python,
  };
}

export function isActiveAnalysisJob(job: WorkflowJob | null) {
  return !!job && ["queued", "running", "cancel_requested"].includes(job.status);
}

export function deriveAgentWorkflowViews(job: WorkflowJob | null): AgentWorkflowStepView[] {
  const events = job?.events ?? [];
  const currentStage = job?.current_stage || events[events.length - 1]?.stage || "";
  const currentIndex = workflowStepIndexForStage(currentStage);
  const lastStartedIndex = lastStartedWorkflowStepIndex(events);
  const active = isActiveAnalysisJob(job);
  const terminalFailed = !!job && ["failed", "canceled", "interrupted"].includes(job.status);
  const failedIndex = terminalFailed ? (currentIndex >= 0 ? currentIndex : lastStartedIndex >= 0 ? lastStartedIndex : 0) : -1;
  const runningIndex = active ? (currentIndex >= 0 ? currentIndex : lastStartedIndex >= 0 ? lastStartedIndex : 0) : -1;

  return AGENT_WORKFLOW_STEPS.map((step, index) => {
    let status: WorkflowNodeStatus = "waiting";
    if (job?.status === "completed") status = "completed";
    else if (failedIndex >= 0) status = index < failedIndex ? "completed" : index === failedIndex ? "failed" : "waiting";
    else if (runningIndex >= 0) status = index < runningIndex ? "completed" : index === runningIndex ? "running" : "waiting";
    const loopMode = job?.agent_mode === "loop";
    const label = loopMode && step.key === "sql" ? "自主分析循环" : loopMode && step.key === "python" ? "证据整合" : loopMode && step.key === "report" ? "报告生成循环" : step.label;
    const detail = loopMode && step.key === "sql" ? "AI 每轮只选择一个白名单工具，观察结果后验证、修复或降级。" : step.detail;
    return { ...step, label, detail, status, events: events.filter((event) => step.stages.includes(event.stage as never)) };
  });
}

export function workflowStepIndexForStage(stage: string): number {
  return AGENT_WORKFLOW_STEPS.findIndex((step) => step.stages.includes(stage as never));
}

function lastStartedWorkflowStepIndex(events: WorkflowEvent[]): number {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const stepIndex = workflowStepIndexForStage(events[index].stage);
    if (stepIndex >= 0) return stepIndex;
  }
  return -1;
}

export function combinedWorkflowStatus(statuses: WorkflowNodeStatus[]): WorkflowNodeStatus {
  if (statuses.includes("failed")) return "failed";
  if (statuses.includes("running")) return "running";
  if (statuses.length && statuses.every((status) => status === "completed")) return "completed";
  return "waiting";
}

export function agentStatusClass(status: WorkflowNodeStatus) {
  return { waiting: "is-waiting", running: "is-running", completed: "is-completed", failed: "is-failed" }[status];
}

export function buildWorkflowLogEntries(job: WorkflowJob): WorkflowLogEntry[] {
  const entries: WorkflowLogEntry[] = [];
  let lastStepIndex = -1;
  const completedSteps = new Set<number>();
  for (const event of job.events) {
    const stepIndex = workflowStepIndexForStage(event.stage);
    if (stepIndex >= 0) {
      for (let index = Math.max(0, lastStepIndex); index < stepIndex; index += 1) {
        if (!completedSteps.has(index)) {
          completedSteps.add(index);
          entries.push({ icon: "✔", text: `${AGENT_WORKFLOW_STEPS[index].logLabel} 完成`, kind: "completed", createdAt: event.created_at });
        }
      }
      lastStepIndex = Math.max(lastStepIndex, stepIndex);
    }
    const failed = event.status === "failed" || event.stage === "failed";
    entries.push({
      icon: failed ? "✖" : stepIndex >= 0 ? "◉" : "•",
      text: translateWorkflowEventMessage(event),
      kind: failed ? "failed" : stepIndex >= 0 ? "running" : "info",
      createdAt: event.created_at,
    });
  }
  if (job.status === "completed") {
    for (let index = Math.max(0, lastStepIndex); index < AGENT_WORKFLOW_STEPS.length; index += 1) {
      if (!completedSteps.has(index)) {
        completedSteps.add(index);
        entries.push({
          icon: "✔",
          text: `${AGENT_WORKFLOW_STEPS[index].logLabel} 完成`,
          kind: "completed",
          createdAt: job.completed_at ?? job.updated_at ?? job.events[job.events.length - 1]?.created_at ?? new Date().toISOString(),
        });
      }
    }
  }
  if (job.error) entries.push({ icon: "✖", text: job.error, kind: "failed", createdAt: job.completed_at ?? job.updated_at ?? new Date().toISOString() });
  return entries;
}

export function translateWorkflowEventMessage(event: WorkflowEvent) {
  const translated: Record<string, string> = {
    "Analysis request accepted.": "分析请求已接收。",
    "Analysis job queued.": "分析任务已进入队列。",
    "Analysis job started.": "分析任务已启动。",
    "Profiling dataset and planning analysis route.": "规划器正在分析数据画像与分析路线...",
    "Designing analysis framework.": "正在解析数据结构并设计分析框架...",
    "Preparing the bounded autonomous analysis loop.": "正在固定自主分析循环的作用域与预算...",
    "Loop scope and budgets fixed by the server.": "已固定工具范围与执行预算。",
    "Loop provider unavailable; using deterministic fallback.": "模型暂不可用，已切换为确定性规则执行。",
    "Deterministic legacy fallback completed.": "确定性规则分析已完成。",
    "Autonomous loop finalized: provider_error.": "自主分析已通过规则路径完成。",
    "AI is selecting the next safe analysis tool.": "AI 正在选择下一项安全分析工具...",
    "Using the deterministic legacy fallback.": "自主循环已切换到确定性规则降级。",
    "Running safe SQL analysis.": "SQL 智能体正在生成并执行安全查询...",
    "Running Python analysis.": "Python 智能体正在执行统计与探索分析...",
    "Preparing iterative analysis rounds.": "正在准备多轮探索分析...",
    "Running foundation analysis round.": "正在执行基础探索轮...",
    "Running parallel exploration round.": "正在执行并行探索轮...",
    "Reflecting on iterative analysis results.": "正在合并多轮探索结论...",
    "Integrating final insights.": "正在整合最终洞察...",
    "Checking evidence, statistical support and analysis grain.": "正在核验数值证据、统计支持和 Join 粒度...",
    "Formatting report charts.": "可视化智能体正在整理图表...",
    "Reviewing analysis quality and gaps.": "审查器正在检查质量与数据缺口...",
    "Generating and saving report.": "报告智能体正在生成并保存报告...",
    "Selecting report generation strategy.": "正在选择报告生成策略。",
    "Generating report draft.": "正在生成报告草稿。",
    "Validating report evidence and chart references.": "正在核验报告证据与图表引用。",
    "Repairing report draft.": "正在修订报告草稿。",
    "Using deterministic report fallback.": "模型报告不可用，正在使用确定性报告模板。",
    "Committing report idempotently.": "正在安全提交报告。",
    "AI is selecting the report strategy from verified evidence.": "正在基于已验证证据选择报告策略。",
    "Generating a traceable report draft.": "正在生成可追溯的报告草稿。",
    "Validating report claims, evidence and chart references.": "正在核验报告结论、证据和图表引用。",
    "Committing the validated report idempotently.": "正在安全提交已验证报告。",
    "Validated report committed idempotently.": "已安全提交报告。",
    "Aggregated provider-reported model token usage.": "已汇总本次模型用量。",
    "Analysis complete.": "完成。",
    "Analysis job completed.": "完成。",
    "Analysis job failed.": "分析任务失败。",
    "Analysis job canceled.": "分析任务已取消。",
  };
  const message = event.message.trim();
  if (translated[message]) return translated[message];
  const selectedTool = message.match(/^Selected tool:\s*([a-z0-9_]+)\.?$/i);
  if (selectedTool) return `已选择“${workflowToolLabel(selectedTool[1])}”工具。`;
  const completedTool = message.match(/^([a-z0-9_]+) succeeded\.?$/i);
  if (completedTool) return `${workflowToolLabel(completedTool[1])}已完成。`;
  const deferredTool = message.match(/^Continuing the planned tool sequence with ([a-z0-9_]+)\.?$/i);
  if (deferredTool) return `正在按顺序执行“${workflowToolLabel(deferredTool[1])}”。`;
  if (/^Model proposed multiple tools;/i.test(message)) return "已将多个工具调用整理为安全的顺序执行。";
  if (/^Deterministic .+ fallback completed\.?$/i.test(message)) return "确定性规则分析已完成。";
  if (/^Produced structured evidence:/i.test(message)) return "已生成结构化分析证据。";
  if (/^Evidence verification:\s*sequence_continues/i.test(message)) return "当前证据有效，正在继续执行既定工具序列。";
  if (/^Evidence verification:\s*need_more_evidence/i.test(message)) return "当前证据尚不充分，正在继续分析。";
  if (/^Evidence verification:\s*(sufficient|passed|validated)/i.test(message)) return "证据验证通过。";
  if (/^Report strategy selected:/i.test(message)) return "已选择报告生成策略。";
  if (/^Report draft revision \d+ generated\.?$/i.test(message)) return "已生成报告草稿。";
  if (/^Report validation outcome:\s*(sufficient|validated|passed)\.?$/i.test(message)) return "报告证据校验通过。";
  if (/^Report validation outcome:/i.test(message)) return "报告证据仍需修订。";
  if (/ completed\.?$/i.test(message)) return `${jobStageLabel(event.stage)}完成。`;
  return `${jobStageLabel(event.stage)}：${message}`;
}

export function workflowToolLabel(toolName: string) {
  return WORKFLOW_TOOL_LABELS[toolName]
    ?? toolName.split("_").filter(Boolean).join(" ")
    ?? "内部分析";
}

export function jobStatusLabel(status: string) {
  return { queued: "排队中", running: "运行中", cancel_requested: "取消中", completed: "已完成", failed: "失败", canceled: "已取消", interrupted: "已中断" }[status] ?? status;
}

export function jobStageLabel(stage: string) {
  return {
    queued: "排队", starting: "启动", planner: "规划器", design_framework: "分析框架", sql_agent: "SQL 智能体", python_agent: "Python 智能体",
    loop_bootstrap: "循环准备", loop_decide: "工具决策", loop_execute: "工具执行", loop_observe: "结果观察", loop_verify: "证据验证", loop_repair: "自动修复", loop_fallback: "规则降级", loop_finalize: "循环收敛", loop_adversarial_repair: "最终返工",
    iterative_prepare_rounds: "准备多轮分析", iterative_round_1: "基础分析轮", iterative_fanout_round: "并行探索轮", iterative_reflect_and_merge: "反思合并",
    integrate_insights: "洞察整合", statistical_verify: "统计审查", adversarial_validate: "质量审查", format_charts: "图表整理",
    agent_loop: "自主分析", model_usage: "模型用量",
    report_agent: "报告生成", report_decide: "报告决策", report_execute: "报告草稿", report_verify: "报告校验", report_repair: "报告修订", report_fallback: "规则报告", report_commit: "报告提交", complete: "完成",
    failed: "失败", canceled: "取消", interrupted: "中断",
  }[stage] ?? stage;
}
