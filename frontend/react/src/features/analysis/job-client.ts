import { API_BASE_URL, apiGet } from "../../api-client";
import type { AnalysisJob } from "../../domain-types";
import { errorMessage } from "../../formatters";
import { isActiveAnalysisJob } from "../../workflow-ui";

export async function pollAnalysisJob(
  jobId: string,
  onUpdate: (job: AnalysisJob) => void,
): Promise<AnalysisJob> {
  const initial = await apiGet<AnalysisJob>(`/analysis/jobs/${jobId}`);
  onUpdate(initial);
  if (!isActiveAnalysisJob(initial)) return initial;
  if (typeof EventSource !== "undefined") {
    try {
      return await streamAnalysisJob(jobId, initial, onUpdate);
    } catch (error) {
      console.warn("Workflow event stream unavailable; falling back to polling.", error);
    }
  }
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const job = await apiGet<AnalysisJob>(`/analysis/jobs/${jobId}`);
    onUpdate(job);
    if (!isActiveAnalysisJob(job)) return job;
    await delay(1000);
  }
  throw new Error("分析任务仍在运行，请稍后在任务列表中查看结果。");
}

export async function analysisErrorMessage(error: unknown) {
  const message = errorMessage(error);
  if (!message.startsWith("无法连接后端服务")) return message;
  try {
    await apiGet("/health");
    return "后端健康检查通过，但分析请求连接中断。请稍后再运行一次；如果持续出现，查看 data/runtime/backend.err.log 中的后端异常。";
  } catch {
    return message;
  }
}

function streamAnalysisJob(
  jobId: string,
  initial: AnalysisJob,
  onUpdate: (job: AnalysisJob) => void,
): Promise<AnalysisJob> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let refreshing = false;
    let terminalEventReceived = false;
    let pollTimer: number | null = null;
    let lastJob = initial;
    const afterSequence = initial.last_event_sequence ?? 0;
    const streamUrl = `${API_BASE_URL}/analysis/jobs/${jobId}/events?after_sequence=${afterSequence}`;
    const source = new EventSource(streamUrl, { withCredentials: true });
    const timeout = window.setTimeout(() => finish(new Error("Workflow event stream timed out.")), 600000);

    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      source.close();
      window.clearTimeout(timeout);
      if (pollTimer !== null) window.clearInterval(pollTimer);
      if (error) reject(error);
      else resolve(lastJob);
    };

    const refreshJob = async () => {
      if (refreshing || settled) return;
      refreshing = true;
      try {
        lastJob = await apiGet<AnalysisJob>(`/analysis/jobs/${jobId}`);
        onUpdate(lastJob);
        if (!isActiveAnalysisJob(lastJob)) finish();
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      } finally {
        refreshing = false;
      }
    };

    source.addEventListener("workflow", () => void refreshJob());
    source.addEventListener("end", () => {
      terminalEventReceived = true;
      source.close();
      void refreshJob();
    });
    source.onerror = () => {
      if (terminalEventReceived) return;
      finish(new Error("Workflow event stream disconnected."));
    };
    pollTimer = window.setInterval(() => void refreshJob(), 2000);
  });
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
