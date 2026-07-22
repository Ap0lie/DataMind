import { CircleCheckBig, Loader2, Square, TriangleAlert } from "lucide-react";
import type { AssistantEvent, AssistantRun } from "./types";

type Props = {
  run: AssistantRun;
  events: AssistantEvent[];
  active: boolean;
  onCancel: () => void;
  onConfirm: (accepted: boolean) => void;
  friendlyError: (error: string) => string;
  eventLabel: (event: AssistantEvent) => string;
};

export function AssistantWorkflowCard({ run, events, active, onCancel, onConfirm, friendlyError, eventLabel }: Props) {
  const visible = events.filter((event) => event.event_type !== "message.delta");
  const latest = [...visible].reverse().find((event) => event.status === "running") ?? visible[visible.length - 1];
  const reportedProgress = visible.reduce((value, event) => {
    const next = Number(event.payload.progress);
    return Number.isFinite(next) ? Math.max(value, next) : value;
  }, 0);
  const completed = visible.filter((event) => event.status === "completed").length;
  const estimatedProgress = visible.length ? Math.round((completed / Math.max(visible.length, completed + 1)) * 100) : 8;
  const progress = run.status === "completed" ? 100 : Math.max(8, Math.min(96, reportedProgress || estimatedProgress));
  const failed = run.status === "failed";
  const awaiting = run.status === "awaiting_confirmation";
  const title = awaiting ? "需要你的确认" : failed ? "这次处理没有完成" : latest ? eventLabel(latest) : "正在准备分析";
  const description = awaiting
    ? "确认后 Kimi 会从当前步骤继续，不会重复执行已完成的工作。"
    : failed
      ? "已保留当前数据与历史结果，你可以稍后重试。"
      : friendlyStage(latest);
  return (
    <section className={`assistant-run-panel ${failed ? "failed" : awaiting ? "confirming" : active ? "running" : "completed"}`}>
      <div className="assistant-run-heading">
        <span className="assistant-run-indicator">{failed ? <TriangleAlert size={18} /> : active ? <Loader2 className="animate-spin" size={18} /> : <CircleCheckBig size={18} />}</span>
        <div><small>{awaiting ? "等待确认" : failed ? "未完成" : "Kimi 正在使用 DataMind"}</small><b>{title}</b><p>{description}</p></div>
        {!awaiting && !failed && <strong>{progress}%</strong>}
        {active && <button type="button" onClick={onCancel}><Square size={13} /> 停止</button>}
      </div>
      {!awaiting && !failed && <div className="assistant-run-progress"><span style={{ width: `${progress}%` }} /></div>}
      {failed && run.error && <div className="assistant-error"><span>{friendlyError(run.error)}</span></div>}
      {awaiting && <div className="assistant-confirm"><p>{run.pending_confirmation.confirmation_type === "soft_delete" ? "该操作会把资产移入回收站，并在 30 天后由系统永久清理。" : "当前分析存在业务歧义，需要你确认后再继续。"}</p><div><button type="button" className="secondary-button" onClick={() => onConfirm(false)}>取消</button><button type="button" className="primary-button" onClick={() => onConfirm(true)}>{run.pending_confirmation.confirmation_type === "soft_delete" ? "确认移入回收站" : "确认并继续"}</button></div></div>}
    </section>
  );
}

function friendlyStage(event?: AssistantEvent) {
  if (!event) return "正在建立安全的数据上下文。";
  if (event.event_type === "retrieval.completed") return "已找到相关数据与报告，正在理解你的需求。";
  if (event.event_type === "analysis.progress") return "正在运行分析、校验结论并生成完整报告。";
  if (event.event_type === "permission.checked") return "权限已确认，正在安全执行你要求的操作。";
  if (event.event_type === "action.planned") return "执行计划已准备，正在开始处理。";
  if (event.event_type === "action.completed" || event.event_type === "tool.completed") return "数据处理已完成，正在整理最终结果。";
  if (event.event_type === "confirmation.required") return "需要确认一个关键选择后才能继续。";
  return "正在处理你的请求，完成后会在这里给出结果。";
}
