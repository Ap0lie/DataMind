import { AlertTriangle, CheckCircle2, RefreshCw, ShieldCheck, Wrench } from "lucide-react";
import { useMemo } from "react";
import type { WorkflowEvent, WorkflowJob } from "./workflow-ui";

const EVENT_LABELS: Record<string, string> = {
  loop_bootstrap: "准备",
  decision: "决策",
  invalid_decision: "重新决策",
  tool_execution: "工具调用",
  observation: "观察",
  verification: "验证",
  repair: "自动修复",
  fallback: "规则降级",
  adversarial_repair: "审查返工",
  loop_finalize: "完成",
  provider_error: "模型降级",
  report_decision: "报告决策",
  report_draft: "生成草稿",
  report_validation: "报告验证",
  report_repair: "报告修订",
  evidence_request: "请求补证据",
  report_fallback: "模板降级",
  report_commit: "提交报告",
};

export function AgentLoopPanel({ job }: { job: WorkflowJob }) {
  const events = useMemo(
    () => job.events.filter((event) => Boolean(event.event_type)),
    [job.events],
  );
  if (job.agent_mode !== "loop") return null;

  const latestBudget = [...events]
    .reverse()
    .map((event) => event.payload?.remaining_budget ?? event.payload?.budget)
    .find((value) => value && typeof value === "object") as Record<string, unknown> | undefined;

  const reportEvents = events.filter((event) => event.event_type?.startsWith("report_") || event.event_type === "evidence_request");
  const analysisEvents = events.filter((event) => !reportEvents.includes(event));
  return (
    <div className="grid gap-3">
    <section className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-4" aria-label="自主分析循环轨迹">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-black text-indigo-950">
            <ShieldCheck size={17} /> 自主分析 Loop
          </div>
          <p className="mt-1 text-xs font-semibold text-indigo-700">只读工具 · 单次单工具 · 有界预算 · 失败自动修复</p>
        </div>
        <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-indigo-700 shadow-sm">
          {job.loop_terminal_reason ? terminalLabel(job.loop_terminal_reason) : `第 ${latestIteration(events)} 轮`}
        </span>
      </div>

      {latestBudget && (
        <div className="mt-3 flex flex-wrap gap-2 text-[11px] font-bold text-indigo-800">
          <BudgetPill label="工具" value={latestBudget.tool_calls_remaining} />
          <BudgetPill label="决策" value={latestBudget.decisions_remaining} />
          <BudgetPill label="Token" value={latestBudget.tokens_remaining} />
        </div>
      )}

      <div className="mt-4 grid gap-2">
        {analysisEvents.map((event, index) => (
          <LoopEventRow key={`${event.sequence ?? event.created_at}-${index}`} event={event} />
        ))}
        {!analysisEvents.length && <div className="rounded-lg bg-white/80 px-3 py-3 text-xs font-semibold text-indigo-500">等待循环事件...</div>}
      </div>
    </section>
    {(reportEvents.length > 0 || job.report_strategy) && (
      <section className="rounded-xl border border-violet-200 bg-violet-50/50 p-4" aria-label="报告生成循环轨迹">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-sm font-black text-violet-950"><ShieldCheck size={17} /> 报告生成循环</div>
            <p className="mt-1 text-xs font-semibold text-violet-700">策略决策 · 数值证据校验 · 最多两次修订 · 一次补分析</p>
          </div>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-violet-700 shadow-sm">
            {job.report_terminal_reason ?? `${job.report_revision_count ?? 0} 次生成`}
          </span>
        </div>
        <div className="mt-4 grid gap-2">
          {reportEvents.map((event, index) => <LoopEventRow key={`${event.sequence ?? event.created_at}-report-${index}`} event={event} />)}
        </div>
      </section>
    )}
    </div>
  );
}

function LoopEventRow({ event }: { event: WorkflowEvent }) {
  const failed = event.status === "failed";
  const repair = event.event_type === "repair" || event.event_type === "adversarial_repair" || event.event_type === "report_repair";
  const fallback = event.event_type === "fallback" || event.event_type === "provider_error" || event.event_type === "report_fallback";
  const Icon = failed || fallback ? AlertTriangle : repair ? Wrench : ["loop_finalize", "report_commit"].includes(event.event_type ?? "") ? CheckCircle2 : RefreshCw;
  return (
    <div className="flex min-w-0 items-start gap-3 rounded-lg border border-indigo-100 bg-white px-3 py-2.5">
      <Icon className={failed || fallback ? "mt-0.5 text-amber-500" : repair ? "mt-0.5 text-violet-500" : "mt-0.5 text-emerald-500"} size={15} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <b className="text-slate-800">{EVENT_LABELS[event.event_type ?? ""] ?? event.event_type}</b>
          {event.iteration != null && <span className="text-slate-400">第 {event.iteration} 轮</span>}
          {event.tool_name && <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-700">{event.tool_name}</code>}
        </div>
        <p className="mt-1 break-words text-xs leading-5 text-slate-600 [overflow-wrap:anywhere]">{event.message}</p>
      </div>
    </div>
  );
}

function BudgetPill({ label, value }: { label: string; value: unknown }) {
  return <span className="rounded-full bg-white px-2.5 py-1 shadow-sm">{label}剩余 {String(value ?? "—")}</span>;
}

function latestIteration(events: WorkflowEvent[]) {
  return events.reduce((highest, event) => Math.max(highest, event.iteration ?? 0), 0);
}

function terminalLabel(reason: string) {
  const labels: Record<string, string> = {
    model_finished: "证据充分",
    evidence_sufficient: "证据充分",
    legacy_fallback: "规则降级",
    provider_error: "模型降级",
    tool_budget_exhausted: "工具预算结束",
    decision_budget_exhausted: "决策预算结束",
    token_budget_exhausted: "Token 预算结束",
    time_budget_exhausted: "时间预算结束",
  };
  return labels[reason] ?? reason;
}
