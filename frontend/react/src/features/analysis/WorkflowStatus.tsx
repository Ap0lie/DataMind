import React, { useEffect, useMemo, useRef } from "react";
import { Loader2 } from "lucide-react";
import { Alert } from "../../components/primitives";
import type { AnalysisJob } from "../../domain-types";
import { formatTime } from "../../formatters";
import {
  AGENT_PLAN_STEPS,
  agentStatusClass,
  buildWorkflowLogEntries,
  combinedWorkflowStatus,
  deriveAgentWorkflowViews,
  isActiveAnalysisJob,
  jobStageLabel,
  jobStatusLabel,
  translateWorkflowEventMessage,
  type AgentWorkflowStepKey,
  type WorkflowNodeStatus,
} from "../../workflow-ui";

export function DynamicAgentPlan({ job }: { job: AnalysisJob | null }) {
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

export function RealtimeWorkflowPanel({ job }: { job: AnalysisJob | null }) {
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

export function AnalysisJobStatusPanel({
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
