export const WORKFLOW_NODE_STATUS = {
  waiting: "waiting",
  running: "running",
  completed: "completed",
  failed: "failed",
} as const;

export const AGENT_WORKFLOW_STEPS = [
  { key: "planner", label: "规划器", logLabel: "Planner", stages: ["queued", "starting", "planner"], detail: "理解问题、检查输入并规划分析路线。" },
  { key: "structure", label: "结构理解", logLabel: "结构理解", stages: ["design_framework", "join_prepare"], detail: "解析字段、画像、数据包关系和分析框架。" },
  { key: "sql", label: "SQL 智能体", logLabel: "SQL Agent", stages: ["sql_agent", "loop_bootstrap", "loop_decide", "loop_execute", "loop_observe", "loop_verify", "loop_repair", "loop_fallback"], detail: "生成安全 SELECT 并在内部 dataset 表上执行。" },
  {
    key: "python",
    label: "Python 智能体",
    logLabel: "Python Agent",
    stages: ["python_agent", "iterative_prepare_rounds", "iterative_round_1", "iterative_fanout_round", "iterative_reflect_and_merge", "loop_finalize", "loop_adversarial_repair", "integrate_insights"],
    detail: "执行统计、探索分析、文本分析和多轮洞察整合。",
  },
  { key: "visualization", label: "可视化智能体", logLabel: "Visualization Agent", stages: ["format_charts"], detail: "整理图表规格、标题和报告图表说明。" },
  { key: "reviewer", label: "审查器", logLabel: "Reviewer", stages: ["adversarial_validate"], detail: "检查分析质量、数据缺口和可追溯性风险。" },
  { key: "report", label: "报告", logLabel: "Report Agent", stages: ["report_agent", "report_decide", "report_execute", "report_verify", "report_repair", "report_fallback", "report_commit", "complete"], detail: "自主选择报告策略、验证证据并幂等提交。" },
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
    "Analysis job started.": "分析任务已启动。",
    "Profiling dataset and planning analysis route.": "Planner 开始分析数据画像与分析路线...",
    "Designing analysis framework.": "正在解析数据结构并设计分析框架...",
    "Preparing the bounded autonomous analysis loop.": "正在固定自主分析循环的作用域与预算...",
    "AI is selecting the next safe analysis tool.": "AI 正在选择下一项安全分析工具...",
    "Using the deterministic legacy fallback.": "自主循环已切换到确定性规则降级。",
    "Running safe SQL analysis.": "SQL 智能体正在生成并执行安全查询...",
    "Running Python analysis.": "Python 智能体正在执行统计与探索分析...",
    "Preparing iterative analysis rounds.": "正在准备多轮探索分析...",
    "Running foundation analysis round.": "正在执行基础探索轮...",
    "Running parallel exploration round.": "正在执行并行探索轮...",
    "Reflecting on iterative analysis results.": "正在合并多轮探索结论...",
    "Integrating final insights.": "正在整合最终洞察...",
    "Formatting report charts.": "可视化智能体正在整理图表...",
    "Reviewing analysis quality and gaps.": "审查器正在检查质量与数据缺口...",
    "Generating and saving report.": "报告智能体正在生成并保存报告...",
    "Analysis complete.": "完成。",
    "Analysis job completed.": "完成。",
    "Analysis job failed.": "分析任务失败。",
    "Analysis job canceled.": "分析任务已取消。",
  };
  return translated[event.message] ?? `${jobStageLabel(event.stage)}：${event.message}`;
}

export function jobStatusLabel(status: string) {
  return { queued: "排队中", running: "运行中", cancel_requested: "取消中", completed: "已完成", failed: "失败", canceled: "已取消", interrupted: "已中断" }[status] ?? status;
}

export function jobStageLabel(stage: string) {
  return {
    queued: "排队", starting: "启动", planner: "规划器", design_framework: "分析框架", sql_agent: "SQL 智能体", python_agent: "Python 智能体",
    loop_bootstrap: "循环准备", loop_decide: "工具决策", loop_execute: "工具执行", loop_observe: "结果观察", loop_verify: "证据验证", loop_repair: "自动修复", loop_fallback: "规则降级", loop_finalize: "循环收敛", loop_adversarial_repair: "最终返工",
    iterative_prepare_rounds: "准备多轮分析", iterative_round_1: "基础分析轮", iterative_fanout_round: "并行探索轮", iterative_reflect_and_merge: "反思合并",
    integrate_insights: "洞察整合", adversarial_validate: "质量审查", format_charts: "图表整理", report_agent: "报告生成", complete: "完成",
    failed: "失败", canceled: "取消", interrupted: "中断",
  }[stage] ?? stage;
}
