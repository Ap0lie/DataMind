import { Activity, CircleCheckBig, Loader2, RefreshCw, TriangleAlert } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { apiGet, apiPost } from "../../api-client";

type DriftStatus = "baseline" | "stable" | "warning" | "critical";

type DriftChange = {
  change_type: string;
  severity: "info" | "warning" | "critical";
  field?: string | null;
  previous_field?: string | null;
  current_field?: string | null;
  previous_value?: unknown;
  current_value?: unknown;
  score?: number | null;
  message: string;
};

type DriftAction = {
  action: string;
  label: string;
  reason: string;
  requires_authorization: boolean;
};

type DatasetDrift = {
  dataset_id: string;
  status: DriftStatus;
  changes: DriftChange[];
  recommended_actions: DriftAction[];
};

type DatasetGroupDrift = {
  group_id: string;
  status: DriftStatus;
  datasets: DatasetDrift[];
  stale_relationship_count: number;
  scanned_at: string;
};

export function DriftMonitorPanel({
  groupId,
  datasetNames = {},
  onScanned,
}: {
  groupId: string;
  datasetNames?: Record<string, string>;
  onScanned?: () => void | Promise<void>;
}) {
  const [result, setResult] = useState<DatasetGroupDrift | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setResult(await apiGet<DatasetGroupDrift>(`/store/dataset-groups/${groupId}/drift`));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法读取数据可靠性状态。");
    } finally {
      setLoading(false);
    }
  }, [groupId]);

  useEffect(() => {
    void load();
  }, [load]);

  const scan = async () => {
    setScanning(true);
    setError("");
    try {
      const next = await apiPost<DatasetGroupDrift>(
        `/store/dataset-groups/${groupId}/drift/scan`,
        {},
        120000,
      );
      setResult(next);
      await onScanned?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "漂移检测失败。");
    } finally {
      setScanning(false);
    }
  };

  if (loading && !result) {
    return (
      <section className="mt-4 flex items-center gap-2 rounded-lg border border-line bg-slate-50 px-4 py-3 text-sm font-bold text-slate-600">
        <Loader2 className="animate-spin" size={16} />
        正在读取数据可靠性状态
      </section>
    );
  }

  const changes = result?.datasets.flatMap((dataset) =>
    dataset.changes.map((change) => ({
      ...change,
      datasetId: dataset.dataset_id,
      datasetName: datasetNames[dataset.dataset_id] ?? dataset.dataset_id,
    })),
  ) ?? [];
  const actions = Array.from(
    new Map(
      (result?.datasets.flatMap((dataset) => dataset.recommended_actions) ?? []).map(
        (action) => [action.action, action],
      ),
    ).values(),
  );
  const warningCount = changes.filter((change) => change.severity === "warning").length;
  const criticalCount = changes.filter((change) => change.severity === "critical").length;
  const displayStatus: DriftStatus = criticalCount > 0
    ? "critical"
    : changes.length > 0
      ? "warning"
      : result?.status ?? "baseline";
  const healthy = displayStatus === "stable" || displayStatus === "baseline";
  const StatusIcon = healthy ? CircleCheckBig : TriangleAlert;

  return (
    <section className="mt-4 rounded-lg border border-line bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className={`rounded-lg p-2 ${healthy ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
            <Activity size={18} />
          </span>
          <div>
            <h5 className="font-black text-slate-950">数据可靠性</h5>
            <p className="mt-1 text-sm font-semibold leading-6 text-slate-600">
              {healthy
                ? "当前 Schema、分布和关系匹配率未发现变化。"
                : `${changes.length} 项待确认变化（${criticalCount} 项严重、${warningCount} 项警告），${result?.stale_relationship_count ?? 0} 条关系需要处理。`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {result && (
            <span className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-2 text-xs font-black ${
              healthy
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : displayStatus === "critical"
                  ? "border-rose-200 bg-rose-50 text-rose-700"
                  : "border-amber-200 bg-amber-50 text-amber-700"
            }`}>
              <StatusIcon size={14} />
              {statusLabel(displayStatus)}
            </span>
          )}
          <button type="button" className="small-button" disabled={scanning} onClick={() => void scan()}>
            {scanning ? <Loader2 className="animate-spin" size={15} /> : <RefreshCw size={15} />}
            {scanning ? "检测中" : "重新检测"}
          </button>
        </div>
      </div>

      {error && <p className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-bold text-rose-700">{error}</p>}

      {!!changes.length && (
        <div className="mt-4 grid max-h-80 gap-2 overflow-y-auto pr-1" aria-label="待确认数据变化">
          {changes.map((change, index) => (
            <div key={`${change.datasetId}-${change.change_type}-${change.field ?? index}`} className="flex items-start gap-2 rounded-lg bg-slate-50 px-3 py-2 text-sm leading-6 text-slate-700">
              <span className={`mt-0.5 rounded px-1.5 py-0.5 text-xs font-black ${
                change.severity === "critical"
                  ? "bg-rose-100 text-rose-700"
                  : change.severity === "warning"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-sky-100 text-sky-700"
              }`}>
                {change.severity === "critical" ? "严重" : change.severity === "warning" ? "警告" : "提示"}
              </span>
              <span>
                <b className="mr-1 text-slate-950">{change.datasetName}</b>
                {driftMessage(change)}
              </span>
            </div>
          ))}
        </div>
      )}

      {!!actions.length && (
        <div className="mt-4 border-t border-line pt-3">
          <p className="text-xs font-black uppercase text-slate-500">建议操作 · 执行前需授权</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {actions.map((action) => (
              <span key={action.action} title={action.reason} className="rounded-lg border border-line bg-white px-3 py-2 text-sm font-bold text-slate-700">
                {action.label}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function statusLabel(status: DriftStatus) {
  if (status === "critical") return "需要处理";
  if (status === "warning") return "存在变化";
  if (status === "baseline") return "基线已建立";
  return "状态稳定";
}

function driftMessage(change: DriftChange) {
  const field = change.field ?? change.current_field ?? change.previous_field ?? "未知字段";
  switch (change.change_type) {
    case "column_renamed":
      return `字段 ${change.previous_field ?? "未知字段"} 可能已重命名为 ${change.current_field ?? field}。`;
    case "column_removed":
      return `字段 ${change.previous_field ?? field} 已被删除。`;
    case "column_added":
      return `新增字段 ${change.current_field ?? field}。`;
    case "type_changed":
      return hasComparisonValues(change)
        ? `字段 ${field} 的类型从 ${displayValue(change.previous_value)} 变为 ${displayValue(change.current_value)}。`
        : localizedFallback(change.message, `字段 ${field} 的类型发生变化。`);
    case "missing_rate_drift":
      return hasComparisonValues(change)
        ? `字段 ${field} 的缺失率从 ${displayRate(change.previous_value)} 变为 ${displayRate(change.current_value)}。`
        : localizedFallback(change.message, `字段 ${field} 的缺失率发生变化。`);
    case "unique_rate_drift":
      return hasComparisonValues(change)
        ? `字段 ${field} 的唯一率从 ${displayRate(change.previous_value)} 变为 ${displayRate(change.current_value)}。`
        : localizedFallback(change.message, `字段 ${field} 的唯一率发生变化。`);
    case "distribution_drift":
      return `字段 ${field} 的均值偏移了 ${Number(change.score ?? 0).toFixed(2)} 个标准差。`;
    case "row_count_drift":
      return `数据行数从 ${displayValue(change.previous_value)} 变为 ${displayValue(change.current_value)}。`;
    default:
      return change.message || "检测到数据变化。";
  }
}

function hasComparisonValues(change: DriftChange) {
  return change.previous_value !== null
    && change.previous_value !== undefined
    && change.current_value !== null
    && change.current_value !== undefined;
}

function localizedFallback(message: string, fallback: string) {
  return /[\u3400-\u9fff]/u.test(message) ? message : fallback;
}

function displayRate(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "未知";
}

function displayValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "未知";
  return typeof value === "number" ? value.toLocaleString("zh-CN") : String(value);
}
