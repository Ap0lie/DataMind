import { AlertTriangle, CheckCircle2, FlaskConical, RefreshCw, ShieldCheck, Wrench } from "lucide-react";
import type { CleaningJob, CleaningJobEvent } from "./api-client";

const LABELS: Record<string, string> = {
  cleaning_bootstrap: "准备范围",
  cleaning_decision: "AI 决策",
  cleaning_execution: "执行策略",
  cleaning_error: "执行反馈",
  cleaning_validation: "质量验证",
  cleaning_repair: "自动修复",
  cleaning_fallback: "规则降级",
  cleaning_commit: "提交版本",
};

export function CleaningLoopPanel({ job }: { job: CleaningJob | null }) {
  if (!job) return null;
  return (
    <section className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50/60 p-4" aria-label="自主清洗循环轨迹">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-black text-emerald-950">
            <ShieldCheck size={17} /> 自主清洗 Loop
          </div>
          <p className="mt-1 text-xs font-semibold text-emerald-700">规则 / LLM / 混合策略 · 沙箱执行 · 质量门禁 · 自动修复</p>
        </div>
        <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-emerald-700 shadow-sm">
          {job.status === "completed" ? `已采用 ${strategyLabel(job.selected_strategy)}` : `${job.progress}% · ${strategyLabel(job.selected_strategy ?? job.cleaning_strategy)}`}
        </span>
      </div>
      <div className="mt-4 grid gap-2">
        {job.events.filter((event) => event.event_type).map((event, index) => (
          <CleaningEventRow key={`${event.sequence ?? event.created_at}-${index}`} event={event} />
        ))}
        {!job.events.some((event) => event.event_type) && (
          <div className="rounded-lg bg-white/80 px-3 py-3 text-xs font-semibold text-emerald-600">等待清洗决策...</div>
        )}
      </div>
      {job.error && <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-xs font-bold text-rose-700">{job.error}</p>}
    </section>
  );
}

function CleaningEventRow({ event }: { event: CleaningJobEvent }) {
  const failed = event.status === "failed" || event.event_type === "cleaning_error";
  const repair = event.event_type === "cleaning_repair";
  const committed = event.event_type === "cleaning_commit";
  const Icon = failed ? AlertTriangle : repair ? Wrench : committed ? CheckCircle2 : event.event_type === "cleaning_validation" ? FlaskConical : RefreshCw;
  return (
    <div className="flex min-w-0 items-start gap-3 rounded-lg border border-emerald-100 bg-white px-3 py-2.5">
      <Icon className={failed ? "mt-0.5 text-amber-500" : repair ? "mt-0.5 text-violet-500" : "mt-0.5 text-emerald-500"} size={15} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <b className="text-slate-800">{LABELS[event.event_type ?? ""] ?? event.event_type}</b>
          {event.strategy && <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-bold text-slate-700">{strategyLabel(event.strategy)}</span>}
          {(event.iteration ?? 0) > 0 && <span className="text-slate-400">第 {event.iteration} 轮</span>}
        </div>
        <p className="mt-1 break-words text-xs leading-5 text-slate-600 [overflow-wrap:anywhere]">{event.message}</p>
      </div>
    </div>
  );
}

function strategyLabel(strategy?: string | null) {
  return { auto: "AI 自动", rules: "本地规则", llm: "LLM", hybrid: "规则 + LLM" }[strategy ?? ""] ?? "待决策";
}
