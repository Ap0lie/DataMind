import { AlertTriangle, CheckCircle2, RefreshCw, ShieldCheck, Wrench } from "lucide-react";
import { useMemo } from "react";
import {
  LOOP_EVENT_LABELS,
  loopAnalysisComponents,
  translateWorkflowEventMessage,
  workflowToolLabel,
  type WorkflowEvent,
  type WorkflowJob,
} from "./workflow-ui";

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
  const componentUsage = loopAnalysisComponents(job);
  const components = componentUsage
    ? [componentUsage.sql ? "sql" : null, componentUsage.python ? "python" : null]
        .filter((component): component is string => component !== null)
    : [];
  const hasExecutionFacts = componentUsage !== null;
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
          {job.loop_terminal_reason
            ? terminalLabel(job.loop_terminal_reason)
            : latestIteration(events) > 0
              ? `第 ${latestIteration(events)} 轮`
              : "准备中"}
        </span>
      </div>

      {latestBudget && (
        <div className="mt-3 flex flex-wrap gap-2 text-[11px] font-bold text-indigo-800">
          <BudgetPill label="工具" value={latestBudget.tool_calls_remaining} />
          <BudgetPill label="决策" value={latestBudget.decisions_remaining} />
          <BudgetPill label="Token" value={latestBudget.tokens_remaining} />
        </div>
      )}

      {job.status === "completed" && hasExecutionFacts && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] font-bold text-indigo-800">
          <span>实际执行</span>
          {components.length
            ? components.map((component) => (
                <span key={component} className="rounded-full bg-white px-2.5 py-1 shadow-sm">
                  {component === "sql" ? "SQL" : component === "python" ? "Python" : component}
                </span>
              ))
            : <span className="rounded-full bg-white px-2.5 py-1 shadow-sm">未调用 SQL/Python</span>}
        </div>
      )}

      <details className="mt-4 rounded-lg border border-indigo-100 bg-white/70 px-3 py-2">
        <summary className="cursor-pointer text-xs font-black text-indigo-800">
          查看循环细节 · {analysisEvents.length} 条
        </summary>
        <div className="mt-3 grid gap-2">
          {analysisEvents.map((event, index) => (
            <LoopEventRow key={`${event.sequence ?? event.created_at}-${index}`} event={event} />
          ))}
          {!analysisEvents.length && <div className="rounded-lg bg-white/80 px-3 py-3 text-xs font-semibold text-indigo-500">等待循环事件...</div>}
        </div>
      </details>
    </section>
    {(reportEvents.length > 0 || job.report_strategy) && (
      <section className="rounded-xl border border-violet-200 bg-violet-50/50 p-4" aria-label="报告生成循环轨迹">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-sm font-black text-violet-950"><ShieldCheck size={17} /> 报告生成循环</div>
            <p className="mt-1 text-xs font-semibold text-violet-700">策略决策 · 数值证据校验 · 最多两次修订 · 一次补分析</p>
          </div>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-violet-700 shadow-sm">
            {job.report_terminal_reason ? terminalLabel(job.report_terminal_reason) : `${job.report_revision_count ?? 0} 次修订`}
          </span>
        </div>
        <details className="mt-4 rounded-lg border border-violet-100 bg-white/70 px-3 py-2">
          <summary className="cursor-pointer text-xs font-black text-violet-800">
            查看报告循环细节 · {reportEvents.length} 条
          </summary>
          <div className="mt-3 grid gap-2">
            {reportEvents.map((event, index) => <LoopEventRow key={`${event.sequence ?? event.created_at}-report-${index}`} event={event} />)}
          </div>
        </details>
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
          <b className="text-slate-800">{LOOP_EVENT_LABELS[event.event_type ?? ""] ?? event.event_type}</b>
          {(event.iteration ?? 0) > 0 && <span className="text-slate-400">第 {event.iteration} 轮</span>}
          {event.tool_name && (
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-bold text-slate-600">
              {workflowToolLabel(event.tool_name)}
            </span>
          )}
        </div>
        <p className="mt-1 break-words text-xs leading-5 text-slate-600 [overflow-wrap:anywhere]">{translateWorkflowEventMessage(event)}</p>
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
    legacy_fallback: "规则执行",
    model_requested_fallback: "模型转入规则执行",
    verification_fallback: "验证后规则执行",
    provider_error: "模型异常后规则执行",
    provider_unavailable: "模型不可用，已转规则执行",
    repeated_duplicate_decision: "重复决策，已转规则执行",
    contract_repair_rejected: "口径修复未通过",
    tool_budget_exhausted: "工具预算结束",
    decision_budget_exhausted: "决策预算结束",
    token_budget_exhausted: "Token 预算结束",
    time_budget_exhausted: "时间预算结束",
    validated: "验证通过",
    committed: "已提交",
    sufficient: "证据充分",
    fallback: "规则执行",
    rules_fallback: "规则生成报告",
    evidence_gap_after_reanalysis: "补充分析后证据仍不足",
  };
  return labels[reason] ?? reason;
}
