import { useEffect, useMemo, useState } from "react";
import { ArchiveRestore, History, Loader2, RotateCcw, ShieldCheck, Undo2, X } from "lucide-react";
import { apiDelete, apiGet, apiPost } from "../../api-client";
import { ASSISTANT_FULL_CAPABILITIES } from "./types";
import type {
  AssistantAction,
  AssistantAssetType,
  AssistantCapability,
  AssistantPermissionGrant,
  AssistantScopeAsset,
  AssistantScopeType,
  RecycledAsset,
} from "./types";

const capabilityLabels: Record<AssistantCapability, string> = {
  data_prepare: "数据准备",
  relationship_manage: "关系管理",
  analysis_manage: "分析任务",
  report_manage: "报告管理",
  semantic_manage: "语义模型",
  asset_recycle: "回收恢复",
};

type Props = {
  open: boolean;
  onClose: () => void;
  onSummaryChange?: (summary: { grants: number; actions: number; recycled: number }) => void;
  scopeType: AssistantScopeType;
  scopeId?: string | null;
  datasets: AssistantScopeAsset[];
  datasetGroups: AssistantScopeAsset[];
  reports: AssistantScopeAsset[];
};

export function AssistantControlPanel({ open, onClose, onSummaryChange, scopeType, scopeId, datasets, datasetGroups, reports }: Props) {
  const [tab, setTab] = useState<"permissions" | "actions" | "recycle">("permissions");
  const [grants, setGrants] = useState<AssistantPermissionGrant[]>([]);
  const [actions, setActions] = useState<AssistantAction[]>([]);
  const [recycled, setRecycled] = useState<RecycledAsset[]>([]);
  const [target, setTarget] = useState("");
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const assets = useMemo(() => [
    ...datasetGroups.map((item) => ({ ...item, type: "dataset_group" as const })),
    ...datasets.map((item) => ({ ...item, type: "dataset" as const })),
    ...reports.map((item) => ({ ...item, type: "report" as const })),
  ], [datasetGroups, datasets, reports]);

  const scopedAsset = useMemo(() => {
    if (scopeType === "auto" || !scopeId) return null;
    return assets.find((item) => item.type === scopeType && item.id === scopeId) ?? null;
  }, [assets, scopeId, scopeType]);
  const effectiveTarget = scopedAsset ? `${scopedAsset.type}:${scopedAsset.id}` : target;
  const effectiveGrant = scopedAsset
    ? grants.find((item) => item.status === "active" && item.asset_type === scopedAsset.type && item.asset_id === scopedAsset.id)
    : null;
  const missingCapabilities = scopedAsset
    ? ASSISTANT_FULL_CAPABILITIES.filter((capability) => !effectiveGrant?.capabilities.includes(capability))
    : ASSISTANT_FULL_CAPABILITIES;
  const fullyAuthorized = Boolean(scopedAsset && missingCapabilities.length === 0);

  useEffect(() => {
    if (scopeType !== "auto") setTarget("");
  }, [scopeId, scopeType]);

  const refresh = async () => {
    const [grantResult, actionResult, recycleResult] = await Promise.allSettled([
      apiGet<{ grants: AssistantPermissionGrant[] }>("/assistant/permission-grants"),
      apiGet<{ actions: AssistantAction[] }>("/assistant/actions?limit=100"),
      apiGet<{ assets: RecycledAsset[] }>("/assistant/recycle-bin"),
    ]);
    const nextGrants = grantResult.status === "fulfilled" ? grantResult.value.grants : grants;
    const nextActions = actionResult.status === "fulfilled" ? actionResult.value.actions : actions;
    const nextRecycled = recycleResult.status === "fulfilled" ? recycleResult.value.assets : recycled;
    if (grantResult.status === "fulfilled") setGrants(nextGrants);
    if (actionResult.status === "fulfilled") setActions(nextActions);
    if (recycleResult.status === "fulfilled") setRecycled(nextRecycled);
    onSummaryChange?.({ grants: nextGrants.length, actions: nextActions.filter((item) => item.status === "running" || (item.reversible && !item.undone_at)).length, recycled: nextRecycled.length });
    const failures = [
      ["权限", grantResult],
      ["操作记录", actionResult],
      ["回收站", recycleResult],
    ].filter((entry) => (entry[1] as PromiseSettledResult<unknown>).status === "rejected") as Array<[string, PromiseRejectedResult]>;
    if (failures.length) {
      throw new Error(failures.map(([label, result]) => `${label}加载失败：${messageOf(result.reason)}`).join("；"));
    }
    setError(null);
  };

  useEffect(() => {
    if (!open) return;
    void refresh().catch((cause) => setError(messageOf(cause)));
  }, [open, scopeId, scopeType]);

  const grant = async () => {
    if (!effectiveTarget) return;
    const [assetType, assetId] = effectiveTarget.split(":", 2) as [AssistantAssetType, string];
    setBusyKey("grant");
    setError(null);
    try {
      await apiPost("/assistant/permission-grants", { asset_type: assetType, asset_id: assetId, capabilities: ASSISTANT_FULL_CAPABILITIES });
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const revoke = async (grantId: string) => {
    setBusyKey(`grant:${grantId}`);
    try {
      await apiDelete(`/assistant/permission-grants/${grantId}`);
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const undo = async (actionId: string) => {
    setBusyKey(`action:${actionId}`);
    try {
      await apiPost(`/assistant/actions/${actionId}/undo`, {});
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const restore = async (asset: RecycledAsset) => {
    setBusyKey(`recycle:${asset.asset_type}:${asset.asset_id}`);
    try {
      await apiPost(`/assistant/recycle-bin/${asset.asset_type}/${asset.asset_id}/restore`, {});
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  if (!open) return null;
  return (
    <>
      <button type="button" className="assistant-control-scrim" aria-label="关闭 Kimi 工作台" onClick={onClose} />
      <aside className="assistant-control-panel" aria-label="Kimi 权限与操作">
        <header>
          <div><ShieldCheck size={19} /><span><b>Kimi 工作台</b><small>授权、操作记录与 30 天回收站</small></span></div>
          <button type="button" className="icon-button" title="关闭" onClick={onClose}><X size={18} /></button>
        </header>
        <nav>
          <button type="button" className={tab === "permissions" ? "active" : ""} onClick={() => setTab("permissions")}><ShieldCheck size={15} /> 权限</button>
          <button type="button" className={tab === "actions" ? "active" : ""} onClick={() => setTab("actions")}><History size={15} /> 操作</button>
          <button type="button" className={tab === "recycle" ? "active" : ""} onClick={() => setTab("recycle")}><ArchiveRestore size={15} /> 回收站</button>
        </nav>
        <div className="assistant-control-content">
          {error && <div className="assistant-control-error">{error}</div>}
          {tab === "permissions" && <>
            <section className="assistant-grant-form">
              <div className="assistant-grant-heading">
                <div><h3>{scopedAsset ? "当前范围权限" : "执行权限"}</h3><p>{scopedAsset ? "读取范围与执行权限已对齐到同一资产。切换顶部范围后，这里会同步更新。" : "自动检索可以读取相关资产；执行修改前仍需选择一个具体资产授权。"}</p></div>
                {scopedAsset && <span className={`assistant-grant-status ${fullyAuthorized ? "complete" : effectiveGrant ? "partial" : "missing"}`}>{fullyAuthorized ? "已完整授权" : effectiveGrant ? `缺少 ${missingCapabilities.length} 项` : "尚未授权"}</span>}
              </div>
              {scopedAsset ? (
                <div className="assistant-current-grant">
                  <div className="assistant-current-grant-asset"><span><ShieldCheck size={17} /></span><div><small>{formatType(scopedAsset.type)} · 当前对话范围</small><b>{scopedAsset.name}</b></div></div>
                  <p>{fullyAuthorized ? "Kimi 可在质量门禁与确认规则内管理此资产。" : effectiveGrant ? `待补全：${missingCapabilities.map((value) => capabilityLabels[value]).join("、")}` : "当前仅可读取；授权后才能清洗、分析、修改报告或管理语义模型。"}</p>
                  <button type="button" disabled={fullyAuthorized || busyKey === "grant"} onClick={() => void grant()}>{busyKey === "grant" ? <Loader2 size={15} className="animate-spin" /> : <ShieldCheck size={15} />} {fullyAuthorized ? "已授权当前范围" : effectiveGrant ? "补全当前范围权限" : "授权当前范围"}</button>
                </div>
              ) : (
                <div className="assistant-auto-grant">
                  <div className="assistant-auto-grant-note"><b>当前为自动检索范围</b><span>问答会自动查找相关资料；如需执行任务，请在此选择资产授权，或先在顶部切换到具体范围。</span></div>
                  <div><select aria-label="选择要授权的资产" value={target} onChange={(event) => setTarget(event.target.value)}><option value="">选择要授权的数据集、数据包或报告</option>{assets.map((item) => <option key={`${item.type}-${item.id}`} value={`${item.type}:${item.id}`}>{formatType(item.type)} · {item.name}</option>)}</select><button type="button" disabled={!target || busyKey === "grant"} onClick={() => void grant()}>{busyKey === "grant" ? <Loader2 size={15} className="animate-spin" /> : <ShieldCheck size={15} />} 授权所选资产</button></div>
                </div>
              )}
            </section>
            <div className="assistant-control-list">{grants.length ? grants.map((item) => <article key={item.grant_id}><div><b>{assetLabel(item.asset_type, item.asset_id, assets)}</b><small>{item.capabilities.map((value) => capabilityLabels[value]).join(" · ")}</small></div><button type="button" disabled={busyKey === `grant:${item.grant_id}`} onClick={() => void revoke(item.grant_id)}>{busyKey === `grant:${item.grant_id}` ? <Loader2 size={13} className="animate-spin" /> : "撤销"}</button></article>) : <EmptyState text="尚未授权 Kimi 管理任何资产。" />}</div>
          </>}
          {tab === "actions" && <div className="assistant-control-list">{actions.length ? actions.map((item) => <article key={item.action_id}><div><b>{actionLabel(item.tool_name)}</b><small>{formatTime(item.created_at)} · {item.status}</small></div>{item.reversible && !item.undone_at && item.status === "completed" ? <button type="button" disabled={busyKey === `action:${item.action_id}`} onClick={() => void undo(item.action_id)}>{busyKey === `action:${item.action_id}` ? <Loader2 size={13} className="animate-spin" /> : <><Undo2 size={13} /> 撤销</>}</button> : <span className="assistant-control-state">{item.undone_at ? "已撤销" : "已记录"}</span>}</article>) : <EmptyState text="Kimi 执行写操作后会在这里留下审计记录。" />}</div>}
          {tab === "recycle" && <div className="assistant-control-list">{recycled.length ? recycled.map((item) => { const key = `recycle:${item.asset_type}:${item.asset_id}`; return <article key={`${item.asset_type}-${item.asset_id}`}><div><b>{item.name}</b><small>{formatType(item.asset_type)} · {formatTime(item.deleted_at)} · {daysLeft(item.purge_after)} 天后清理</small></div><button type="button" disabled={busyKey === key} onClick={() => void restore(item)}>{busyKey === key ? <Loader2 size={13} className="animate-spin" /> : <><RotateCcw size={13} /> 恢复</>}</button></article>; }) : <EmptyState text="回收站为空。软删除资产会在此保留 30 天。" />}</div>}
        </div>
      </aside>
    </>
  );
}

function EmptyState({ text }: { text: string }) { return <div className="assistant-control-empty">{text}</div>; }
function messageOf(error: unknown) { return error instanceof Error ? error.message : String(error); }
function formatTime(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }); }
function daysLeft(value: string) { return Math.max(0, Math.ceil((new Date(value).getTime() - Date.now()) / 86_400_000)); }
function formatType(value: AssistantAssetType) { return { dataset: "数据集", dataset_group: "数据包", report: "报告", semantic_model: "语义模型" }[value]; }
function assetLabel(type: AssistantAssetType, id: string, assets: Array<AssistantScopeAsset & { type: AssistantAssetType }>) { return assets.find((item) => item.type === type && item.id === id)?.name ?? `${formatType(type)} ${id.slice(0, 8)}`; }
function actionLabel(value: string) { return ({ start_cleaning: "启动清洗", activate_cleaning_version: "切换清洗版本", rollback_cleaning_version: "回滚清洗版本", update_column_metadata: "更新字段元数据", save_relationship_plan: "保存数据关系", start_analysis: "启动分析", cancel_analysis: "取消分析", retry_analysis: "重试分析", rename_report: "重命名报告", create_semantic_draft: "创建语义草稿", update_semantic_draft: "更新语义草稿", publish_semantic_model: "发布语义模型", soft_delete_asset: "移入回收站", restore_asset: "恢复资产" } as Record<string, string>)[value] ?? value; }
