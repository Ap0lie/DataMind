const AUTH_STORAGE_KEY = "datamind.authUser.v1";

export const API_BASE_URL =
  import.meta.env.VITE_DATAMIND_API_BASE_URL ?? "http://127.0.0.1:8010/api/v1";
const API_FALLBACK_BASE_URL = localApiFallback(API_BASE_URL);

export type AuthUser = {
  user_id: string;
  display_name: string;
  csrf_token?: string | null;
  expires_at?: string | null;
};

export async function apiGet<T>(path: string): Promise<T> {
  try {
    const response = await fetchApi(path);
    if (!response.ok) throw new Error(await readableError(response));
    return (await response.json()) as T;
  } catch (error) {
    throw normalizeFetchError(error);
  }
}

export function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetchApi(path, init);
}

export async function apiPost<T>(path: string, payload: unknown, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchApi(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await readableError(response));
    return (await response.json()) as T;
  } catch (error) {
    throw normalizeFetchError(error);
  } finally {
    window.clearTimeout(timer);
  }
}

export async function apiPostForm<T>(path: string, formData: FormData, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchApi(path, { method: "POST", body: formData, signal: controller.signal });
    if (!response.ok) throw new Error(await readableError(response));
    return (await response.json()) as T;
  } catch (error) {
    throw normalizeFetchError(error);
  } finally {
    window.clearTimeout(timer);
  }
}

export async function apiPatch<T>(path: string, payload: unknown, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchApi(path, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await readableError(response));
    return (await response.json()) as T;
  } catch (error) {
    throw normalizeFetchError(error);
  } finally {
    window.clearTimeout(timer);
  }
}

export async function apiPut<T>(path: string, payload: unknown, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchApi(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await readableError(response));
    return (await response.json()) as T;
  } catch (error) {
    throw normalizeFetchError(error);
  } finally {
    window.clearTimeout(timer);
  }
}

export async function apiDelete(path: string): Promise<void> {
  try {
    const response = await fetchApi(path, { method: "DELETE" });
    if (!response.ok) throw new Error(await readableError(response));
  } catch (error) {
    throw normalizeFetchError(error);
  }
}

export async function logoutSession(): Promise<void> {
  const response = await fetchApi("/auth/logout", { method: "POST" });
  if (!response.ok && response.status !== 401) throw new Error(await readableError(response));
}

export type CleaningJobEvent = {
  sequence?: number;
  stage: string;
  status: string;
  progress: number;
  message: string;
  event_type?: string | null;
  iteration?: number | null;
  strategy?: string | null;
  payload?: Record<string, unknown>;
  created_at: string;
};

export type CleaningJob = {
  job_id: string;
  dataset_id: string;
  requirement: string;
  cleaning_strategy: "auto" | "rules" | "llm" | "hybrid";
  selected_strategy?: "rules" | "llm" | "hybrid" | null;
  status: string;
  progress: number;
  current_stage: string;
  events: CleaningJobEvent[];
  loop_summary?: Record<string, unknown>;
  terminal_reason?: string | null;
  error?: string | null;
  cleaning_run_id?: string | null;
  last_event_sequence?: number;
};

export async function runDatasetCleaning(
  datasetId: string,
  options: {
    requirement: string;
    strategy?: "auto" | "rules" | "llm" | "hybrid";
    onJob?: (job: CleaningJob) => void;
  },
): Promise<{ preview_records: Record<string, unknown>[] }> {
  const job = await apiPost<CleaningJob>(`/store/datasets/${datasetId}/cleaning-jobs`, {
    requirement: options.requirement,
    cleaning_strategy: options.strategy ?? "auto",
  });
  options.onJob?.(job);
  const completed = await pollCleaningJob(datasetId, job, options.onJob);
  if (completed.status !== "completed") {
    throw new Error(completed.error || "清洗任务未成功完成，原活动版本保持不变。");
  }
  return apiGet(`/store/datasets/${datasetId}/cleaning-jobs/${completed.job_id}/result`);
}

async function pollCleaningJob(
  datasetId: string,
  initial: CleaningJob,
  onJob?: (job: CleaningJob) => void,
): Promise<CleaningJob> {
  if (!isActiveCleaningJob(initial)) return initial;
  if (typeof EventSource !== "undefined") {
    try {
      return await streamCleaningJob(datasetId, initial, onJob);
    } catch (error) {
      console.warn("Cleaning event stream unavailable; falling back to polling.", error);
    }
  }
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const job = await apiGet<CleaningJob>(`/store/datasets/${datasetId}/cleaning-jobs/${initial.job_id}`);
    onJob?.(job);
    if (!isActiveCleaningJob(job)) return job;
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  throw new Error("清洗任务仍在运行，请稍后查看任务轨迹。");
}

function streamCleaningJob(
  datasetId: string,
  initial: CleaningJob,
  onJob?: (job: CleaningJob) => void,
): Promise<CleaningJob> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let refreshing = false;
    let lastJob = initial;
    const cursor = initial.last_event_sequence ?? 0;
    const source = new EventSource(
      `${API_BASE_URL}/store/datasets/${datasetId}/cleaning-jobs/${initial.job_id}/events?after_sequence=${cursor}`,
      { withCredentials: true },
    );
    const timeout = window.setTimeout(() => finish(new Error("Cleaning event stream timed out.")), 240000);
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      source.close();
      window.clearTimeout(timeout);
      error ? reject(error) : resolve(lastJob);
    };
    const refresh = async () => {
      if (refreshing || settled) return;
      refreshing = true;
      try {
        lastJob = await apiGet<CleaningJob>(`/store/datasets/${datasetId}/cleaning-jobs/${initial.job_id}`);
        onJob?.(lastJob);
        if (!isActiveCleaningJob(lastJob)) finish();
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      } finally {
        refreshing = false;
      }
    };
    source.addEventListener("cleaning", () => void refresh());
    source.addEventListener("end", () => void refresh());
    source.onerror = () => finish(new Error("Cleaning event stream disconnected."));
  });
}

function isActiveCleaningJob(job: CleaningJob) {
  return ["queued", "running", "cancel_requested"].includes(job.status);
}

export function loadAuthUser(): AuthUser | null {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<AuthUser>;
    if (typeof parsed.user_id !== "string" || typeof parsed.display_name !== "string") return null;
    return {
      user_id: parsed.user_id,
      display_name: parsed.display_name,
      csrf_token: typeof parsed.csrf_token === "string" ? parsed.csrf_token : null,
      expires_at: typeof parsed.expires_at === "string" ? parsed.expires_at : null,
    };
  } catch (error) {
    console.warn("Failed to restore auth user.", error);
    return null;
  }
}

export function saveAuthUser(user: AuthUser | null) {
  try {
    if (!user) window.localStorage.removeItem(AUTH_STORAGE_KEY);
    else window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(user));
  } catch (error) {
    console.warn("Failed to persist auth user.", error);
  }
}

async function fetchApi(path: string, init?: RequestInit) {
  const requestInit = withAuthHeader(init);
  try {
    return await fetch(`${API_BASE_URL}${path}`, requestInit);
  } catch (error) {
    if (!isFetchNetworkError(error) || !API_FALLBACK_BASE_URL || requestInit.signal?.aborted) throw error;
    return fetch(`${API_FALLBACK_BASE_URL}${path}`, requestInit);
  }
}

function withAuthHeader(init?: RequestInit): RequestInit {
  const user = loadAuthUser();
  const headers = new Headers(init?.headers);
  if (user?.user_id) headers.set("X-DataMind-User", user.user_id);
  const method = String(init?.method ?? "GET").toUpperCase();
  if (user?.csrf_token && !["GET", "HEAD", "OPTIONS"].includes(method)) headers.set("X-CSRF-Token", user.csrf_token);
  return { ...(init ?? {}), headers, credentials: "include" };
}

function localApiFallback(baseUrl: string) {
  if (baseUrl.includes("127.0.0.1")) return baseUrl.replace("127.0.0.1", "localhost");
  if (baseUrl.includes("localhost")) return baseUrl.replace("localhost", "127.0.0.1");
  return null;
}

function isFetchNetworkError(error: unknown) {
  return error instanceof TypeError && error.message.toLowerCase().includes("failed to fetch");
}

async function readableError(response: Response) {
  const text = await response.text();
  if (!text) return `接口错误 ${response.status}`;
  try {
    const payload = JSON.parse(text) as { detail?: unknown; message?: unknown };
    const detail = payload.detail ?? payload.message;
    if (typeof detail === "string") return `接口错误 ${response.status}: ${detail}`;
    return `接口错误 ${response.status}: ${JSON.stringify(detail ?? payload)}`;
  } catch {
    return `接口错误 ${response.status}: ${text}`;
  }
}

function normalizeFetchError(error: unknown) {
  if (error instanceof DOMException && error.name === "AbortError") {
    return new Error("请求超时：本次操作耗时较长，请稍后重试，或确认后端服务仍在运行。");
  }
  if (isFetchNetworkError(error)) {
    const fallbackText = API_FALLBACK_BASE_URL ? `；也已尝试备用地址 ${API_FALLBACK_BASE_URL}` : "";
    return new Error(`无法连接后端服务：请确认 FastAPI 已启动，并且前端 API 地址为 ${API_BASE_URL}${fallbackText}。`);
  }
  return error;
}
