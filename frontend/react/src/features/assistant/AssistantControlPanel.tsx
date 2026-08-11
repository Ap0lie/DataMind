import { useEffect, useMemo, useState } from "react";
import { Activity, ArchiveRestore, Brain, Check, History, Loader2, Pencil, Pin, PinOff, RotateCcw, ShieldCheck, Trash2, Undo2, X } from "lucide-react";
import { apiDelete, apiGet, apiPatch, apiPost } from "../../api-client";
import { assistantCapabilitiesForAsset } from "./types";
import type {
  AssistantAction,
  AssistantAssetType,
  AssistantCapability,
  AssistantMemory,
  AssistantMemoryEffectiveness,
  AssistantMemoryStatus,
  AssistantMemoryType,
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

const memoryTypeLabels: Record<AssistantMemoryType, string> = {
  preference: "个人偏好",
  terminology: "业务术语",
  metric_definition: "指标口径",
  business_context: "业务背景",
  workflow_preference: "工作流偏好",
  analysis_experience: "分析经验",
};

type Props = {
  open: boolean;
  initialTab?: "permissions" | "actions" | "memory" | "recycle";
  onClose: () => void;
  onSummaryChange?: (summary: { grants: number; actions: number; recycled: number; memories: number }) => void;
  onAssetsChanged?: () => void | Promise<void>;
  refreshToken?: number;
  scopeType: AssistantScopeType;
  scopeId?: string | null;
  datasets: AssistantScopeAsset[];
  datasetGroups: AssistantScopeAsset[];
  reports: AssistantScopeAsset[];
};

export function AssistantControlPanel({ open, initialTab = "permissions", onClose, onSummaryChange, onAssetsChanged, refreshToken, scopeType, scopeId, datasets, datasetGroups, reports }: Props) {
  const [tab, setTab] = useState<"permissions" | "actions" | "memory" | "recycle">("permissions");
  const [grants, setGrants] = useState<AssistantPermissionGrant[]>([]);
  const [actions, setActions] = useState<AssistantAction[]>([]);
  const [recycled, setRecycled] = useState<RecycledAsset[]>([]);
  const [memories, setMemories] = useState<AssistantMemory[]>([]);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryType, setMemoryType] = useState<AssistantMemoryType | "all">("all");
  const [memoryStatus, setMemoryStatus] = useState<AssistantMemoryStatus | "all">("all");
  const [memoryScope, setMemoryScope] = useState<"all" | "user" | "asset">("all");
  const [memoryView, setMemoryView] = useState<"memory" | "experience" | "quality" | "history">("memory");
  const [memoryEffectiveness, setMemoryEffectiveness] = useState<AssistantMemoryEffectiveness | null>(null);
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [editingMemory, setEditingMemory] = useState<{ id: string; content: string; memoryType: AssistantMemoryType } | null>(null);
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
  const targetAssetType = scopedAsset?.type ?? (
    effectiveTarget ? effectiveTarget.split(":", 1)[0] as AssistantAssetType : null
  );
  const targetCapabilities = targetAssetType
    ? assistantCapabilitiesForAsset(targetAssetType)
    : [];
  const missingCapabilities = scopedAsset
    ? targetCapabilities.filter((capability) => !effectiveGrant?.capabilities.includes(capability))
    : targetCapabilities;
  const fullyAuthorized = Boolean(scopedAsset && missingCapabilities.length === 0);
  const filteredMemories = useMemo(() => memories.filter((item) => {
    const matchesQuery = !memoryQuery.trim() || item.content.toLocaleLowerCase().includes(memoryQuery.trim().toLocaleLowerCase());
    const matchesType = memoryType === "all" || item.memory_type === memoryType;
    const matchesStatus = memoryStatus === "all" || item.status === memoryStatus;
    const matchesScope = memoryScope === "all" || (memoryScope === "user" ? item.scope_type === "user" : item.scope_type !== "user");
    const matchesView = memoryView === "experience"
      ? item.memory_kind === "episodic"
      : memoryView === "quality"
        ? item.status === "dormant" || item.utility_score < 0.35 || !item.last_used_at
      : memoryView === "history"
        ? ["superseded", "stale", "dormant", "recycled"].includes(item.status)
        : item.memory_kind === "semantic" && !["superseded", "stale"].includes(item.status);
    return matchesQuery && matchesType && matchesStatus && matchesScope && matchesView;
  }), [memories, memoryQuery, memoryScope, memoryStatus, memoryType, memoryView]);

  useEffect(() => {
    if (scopeType !== "auto") setTarget("");
  }, [scopeId, scopeType]);

  useEffect(() => {
    if (open) setTab(initialTab);
  }, [initialTab, open]);

  const refresh = async () => {
    const [grantResult, actionResult, memoryResult, memorySettingsResult, effectivenessResult, recycleResult] = await Promise.allSettled([
      apiGet<{ grants: AssistantPermissionGrant[] }>("/assistant/permission-grants"),
      apiGet<{ actions: AssistantAction[] }>("/assistant/actions?limit=100"),
      apiGet<{ memories: AssistantMemory[] }>("/assistant/memories?limit=500"),
      apiGet<{ enabled: boolean }>("/assistant/memory-settings"),
      apiGet<AssistantMemoryEffectiveness>("/assistant/memory-effectiveness"),
      apiGet<{ assets: RecycledAsset[] }>("/assistant/recycle-bin"),
    ]);
    const nextGrants = grantResult.status === "fulfilled" ? grantResult.value.grants : grants;
    const nextActions = actionResult.status === "fulfilled" ? actionResult.value.actions : actions;
    const nextMemories = memoryResult.status === "fulfilled" ? memoryResult.value.memories : memories;
    const nextRecycled = recycleResult.status === "fulfilled" ? recycleResult.value.assets : recycled;
    if (grantResult.status === "fulfilled") setGrants(nextGrants);
    if (actionResult.status === "fulfilled") setActions(nextActions);
    if (memoryResult.status === "fulfilled") setMemories(nextMemories);
    if (memorySettingsResult.status === "fulfilled") setMemoryEnabled(memorySettingsResult.value.enabled);
    if (effectivenessResult.status === "fulfilled") setMemoryEffectiveness(effectivenessResult.value);
    if (recycleResult.status === "fulfilled") setRecycled(nextRecycled);
    onSummaryChange?.({ grants: nextGrants.length, actions: nextActions.filter((item) => item.status === "running" || (item.reversible && !item.undone_at)).length, recycled: nextRecycled.length, memories: nextMemories.filter((item) => item.status !== "recycled").length });
    const failures = [
      ["权限", grantResult],
      ["操作记录", actionResult],
      ["记忆", memoryResult],
      ["记忆设置", memorySettingsResult],
      ["记忆质量", effectivenessResult],
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
  }, [open, refreshToken, scopeId, scopeType]);

  const grant = async () => {
    if (!effectiveTarget) return;
    const [assetType, assetId] = effectiveTarget.split(":", 2) as [AssistantAssetType, string];
    setBusyKey("grant");
    setError(null);
    try {
      await apiPost("/assistant/permission-grants", {
        asset_type: assetType,
        asset_id: assetId,
        capabilities: assistantCapabilitiesForAsset(assetType),
      });
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
      await Promise.all([refresh(), onAssetsChanged?.()]);
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
      await Promise.all([refresh(), onAssetsChanged?.()]);
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const confirmMemory = async (memoryId: string) => {
    setBusyKey(`memory:${memoryId}`);
    try {
      await apiPost(`/assistant/memories/${memoryId}/confirm`, {});
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const saveMemory = async () => {
    if (!editingMemory) return;
    setBusyKey(`memory:${editingMemory.id}`);
    try {
      await apiPatch(`/assistant/memories/${editingMemory.id}`, { content: editingMemory.content, memory_type: editingMemory.memoryType });
      setEditingMemory(null);
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const toggleMemoryPin = async (memory: AssistantMemory) => {
    setBusyKey(`memory:${memory.memory_id}`);
    try {
      await apiPatch(`/assistant/memories/${memory.memory_id}`, { pinned: !memory.pinned });
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const recycleMemory = async (memory: AssistantMemory) => {
    setBusyKey(`memory:${memory.memory_id}`);
    try {
      await apiDelete(`/assistant/memories/${memory.memory_id}`);
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const restoreMemory = async (memory: AssistantMemory) => {
    setBusyKey(`memory:${memory.memory_id}`);
    try {
      await apiPost(`/assistant/memories/${memory.memory_id}/restore`, {});
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const reactivateMemory = async (memory: AssistantMemory) => {
    setBusyKey(`memory:${memory.memory_id}`);
    try {
      await apiPost(`/assistant/memories/${memory.memory_id}/reactivate`, {});
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const wakeMemory = async (memory: AssistantMemory) => {
    setBusyKey(`memory:${memory.memory_id}`);
    try {
      await apiPost(`/assistant/memories/${memory.memory_id}/wake`, {});
      await refresh();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusyKey(null);
    }
  };

  const toggleMemoryEnabled = async () => {
    setBusyKey("memory-settings");
    try {
      const result = await apiPatch<{ enabled: boolean }>("/assistant/memory-settings", { enabled: !memoryEnabled });
      setMemoryEnabled(result.enabled);
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
          <div><ShieldCheck size={19} /><span><b>Kimi 工作台</b><small>授权、记忆、操作记录与回收站</small></span></div>
          <button type="button" className="icon-button" title="关闭" onClick={onClose}><X size={18} /></button>
        </header>
        <nav>
          <button type="button" className={tab === "permissions" ? "active" : ""} onClick={() => setTab("permissions")}><ShieldCheck size={15} /> 权限</button>
          <button type="button" className={tab === "actions" ? "active" : ""} onClick={() => setTab("actions")}><History size={15} /> 操作</button>
          <button type="button" className={tab === "memory" ? "active" : ""} onClick={() => setTab("memory")}><Brain size={15} /> 记忆</button>
          <button type="button" className={tab === "recycle" ? "active" : ""} onClick={() => setTab("recycle")}><ArchiveRestore size={15} /> 回收站</button>
        </nav>
        <div className="assistant-control-content">
          {error && <div className="assistant-control-error">{error}</div>}
          {tab === "permissions" && <>
            <section className="assistant-grant-form">
              <div className="assistant-grant-heading">
                <div><h3>{scopedAsset ? "当前范围权限" : "执行权限"}</h3><p>{scopedAsset ? "读取范围与执行权限已对齐到同一资产。切换顶部范围后，这里会同步更新。" : "自动检索可以读取相关资产；执行修改前仍需选择一个具体资产授权。"}</p></div>
                {scopedAsset && <span className={`assistant-grant-status ${fullyAuthorized ? "complete" : effectiveGrant ? "partial" : "missing"}`}>{fullyAuthorized ? "当前范围权限已齐备" : effectiveGrant ? `缺少 ${missingCapabilities.length} 项` : "尚未授权"}</span>}
              </div>
              {scopedAsset ? (
                <div className="assistant-current-grant">
                  <div className="assistant-current-grant-asset"><span><ShieldCheck size={17} /></span><div><small>{formatType(scopedAsset.type)} · 当前对话范围</small><b>{scopedAsset.name}</b></div></div>
                  <p>{fullyAuthorized ? "Kimi 可使用该资产类型允许的全部能力，并继续受质量门禁与确认规则约束。" : effectiveGrant ? `待补全：${missingCapabilities.map((value) => capabilityLabels[value]).join("、")}` : "当前仅可读取；授权后才能使用该资产类型允许的执行能力。"}</p>
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
          {tab === "memory" && <>
            <div className="assistant-memory-settings">
              <div><b>长期记忆</b><span>{memoryEnabled ? "Kimi 会召回相关偏好和已验证经验" : "已停止长期记忆的读取与写入；对话摘要仍保留"}</span></div>
              <button type="button" role="switch" aria-checked={memoryEnabled} disabled={busyKey === "memory-settings"} className={memoryEnabled ? "active" : ""} onClick={() => void toggleMemoryEnabled()}>{busyKey === "memory-settings" ? <Loader2 size={14} className="animate-spin" /> : memoryEnabled ? "已启用" : "已关闭"}</button>
            </div>
            <div className="assistant-memory-views" role="tablist" aria-label="记忆视图">
              <button type="button" className={memoryView === "memory" ? "active" : ""} onClick={() => setMemoryView("memory")}>用户记忆</button>
              <button type="button" className={memoryView === "experience" ? "active" : ""} onClick={() => setMemoryView("experience")}>分析经验</button>
              <button type="button" className={memoryView === "quality" ? "active" : ""} onClick={() => setMemoryView("quality")}>质量</button>
              <button type="button" className={memoryView === "history" ? "active" : ""} onClick={() => setMemoryView("history")}>版本历史</button>
            </div>
            {memoryView === "quality" && memoryEffectiveness && <div className="assistant-memory-quality">
              <header><div><Activity size={16} /><b>记忆有效性</b></div><span>{memoryEffectiveness.shadow_mode ? "影子评分" : "自动休眠已启用"}</span></header>
              <div>
                <article><b>{Math.round(memoryEffectiveness.average_utility * 100)}%</b><span>平均效用</span></article>
                <article><b>{memoryEffectiveness.selected_usage_count}</b><span>实际采用</span></article>
                <article><b>{memoryEffectiveness.low_quality_memories}</b><span>低质量</span></article>
                <article><b>{memoryEffectiveness.dormant_memories}</b><span>已休眠</span></article>
              </div>
              <p>有用 {memoryEffectiveness.feedback_counts.helpful ?? 0} · 无关 {memoryEffectiveness.feedback_counts.irrelevant ?? 0} · 错误 {memoryEffectiveness.feedback_counts.wrong ?? 0} · 从未使用 {memoryEffectiveness.never_used_memories}</p>
              <p>未采用：相关性不足 {memoryEffectiveness.suppression_counts.below_relevance_threshold ?? 0} · 去重或超出条数 {memoryEffectiveness.suppression_counts.mmr_or_limit ?? 0} · 超出上下文预算 {memoryEffectiveness.suppression_counts.context_budget ?? 0}</p>
            </div>}
            <div className="assistant-memory-toolbar">
              <input aria-label="搜索记忆" placeholder="搜索偏好、术语或指标口径" value={memoryQuery} onChange={(event) => setMemoryQuery(event.target.value)} />
              <select aria-label="记忆类型" value={memoryType} onChange={(event) => setMemoryType(event.target.value as AssistantMemoryType | "all")}><option value="all">全部类型</option>{Object.entries(memoryTypeLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
              <select aria-label="记忆状态" value={memoryStatus} onChange={(event) => setMemoryStatus(event.target.value as AssistantMemoryStatus | "all")}><option value="all">全部状态</option><option value="active">使用中</option><option value="pending">待确认</option><option value="superseded">已替代</option><option value="stale">已失效</option><option value="dormant">已休眠</option><option value="recycled">回收中</option></select>
              <select aria-label="记忆范围" value={memoryScope} onChange={(event) => setMemoryScope(event.target.value as "all" | "user" | "asset")}><option value="all">全部范围</option><option value="user">全局</option><option value="asset">资产</option></select>
            </div>
            <div className="assistant-memory-list">{filteredMemories.length ? filteredMemories.map((item) => {
              const busy = busyKey === `memory:${item.memory_id}`;
              const editing = editingMemory?.id === item.memory_id;
              return <article key={item.memory_id} className={`assistant-memory-item ${item.status}`}>
                {editing ? <div className="assistant-memory-editor"><select value={editingMemory.memoryType} onChange={(event) => setEditingMemory({ ...editingMemory, memoryType: event.target.value as AssistantMemoryType })}>{Object.entries(memoryTypeLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><textarea value={editingMemory.content} maxLength={4000} onChange={(event) => setEditingMemory({ ...editingMemory, content: event.target.value })} /><div><button type="button" onClick={() => setEditingMemory(null)}>取消</button><button type="button" disabled={!editingMemory.content.trim() || busy} onClick={() => void saveMemory()}>{busy ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />} 保存</button></div></div> : <>
                  <div className="assistant-memory-head"><span>{memoryTypeLabels[item.memory_type]}</span><span>{memoryScopeLabel(item)}</span><span>v{item.version}</span>{item.pinned && <Pin size={13} aria-label="已固定" />}</div>
                  <p>{item.content}</p>
                  <small>{memoryStatusLabel(item.status)} · {item.application_policy === "always" ? "持续适用" : "相关时使用"} · {formatTime(item.updated_at)}{item.last_used_at ? ` · 最近使用 ${formatTime(item.last_used_at)}` : ""}{item.source_conversation_deleted ? " · 来源对话已删除" : ""}</small>
                  <div className="assistant-memory-score"><span><b>{Math.round(item.utility_score * 100)}%</b> 效用</span><span>有用 {item.helpful_count}</span><span>无关 {item.irrelevant_count}</span><span>错误 {item.wrong_count}</span>{item.validated_reuse_count > 0 && <span>验证复用 {item.validated_reuse_count}</span>}</div>
                  <small>主题：{item.entity_key} · {item.predicate}{item.unit ? ` · 单位 ${item.unit}` : ""}</small>
                  {item.status === "stale" && typeof item.structured_value.stale_reason === "string" && <small className="assistant-memory-stale-reason">失效原因：{item.structured_value.stale_reason}</small>}
                  {item.status === "dormant" && item.dormant_reason && <small className="assistant-memory-stale-reason">休眠原因：{item.dormant_reason}</small>}
                  <div className="assistant-memory-actions">
                    {item.status === "pending" && <button type="button" disabled={busy} onClick={() => void confirmMemory(item.memory_id)}><Check size={14} /> 确认</button>}
                    {item.memory_kind === "semantic" && ["active", "pending"].includes(item.status) && <button type="button" disabled={busy} onClick={() => setEditingMemory({ id: item.memory_id, content: item.content, memoryType: item.memory_type })}><Pencil size={14} /> 编辑</button>}
                    {item.memory_kind === "semantic" && item.status === "active" && <button type="button" disabled={busy} onClick={() => void toggleMemoryPin(item)}>{item.pinned ? <PinOff size={14} /> : <Pin size={14} />}{item.pinned ? "取消固定" : "固定"}</button>}
                    {["superseded", "stale"].includes(item.status) && <button type="button" disabled={busy} onClick={() => void reactivateMemory(item)}><RotateCcw size={14} /> 重新启用</button>}
                    {item.status === "dormant" && <button type="button" disabled={busy} onClick={() => void wakeMemory(item)}><RotateCcw size={14} /> 唤醒</button>}
                    {item.status === "recycled" ? <button type="button" disabled={busy} onClick={() => void restoreMemory(item)}><RotateCcw size={14} /> 恢复</button> : <button type="button" disabled={busy} onClick={() => void recycleMemory(item)}><Trash2 size={14} />{item.status === "pending" ? "忽略" : "回收"}</button>}
                  </div>
                </>}
              </article>;
            }) : <EmptyState text={memoryView === "experience" ? "还没有通过统计审查并可复用的分析经验。" : memoryView === "quality" ? "当前没有低质量、休眠或从未使用的记忆。" : memoryView === "history" ? "尚无被替代、失效、休眠或回收的历史版本。" : "没有符合筛选条件的长期记忆。明确说“请记住……”后，Kimi 会把它保存在这里。"} />}</div>
          </>}
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
function memoryScopeLabel(memory: AssistantMemory) { return memory.scope_type === "user" ? "全局" : `${formatType(memory.scope_type)} · ${memory.scope_id?.slice(0, 8) ?? "-"}`; }
function memoryStatusLabel(value: AssistantMemoryStatus) { return { active: "使用中", pending: "待确认", superseded: "已替代", stale: "已失效", dormant: "已休眠", recycled: "回收中" }[value]; }
function assetLabel(type: AssistantAssetType, id: string, assets: Array<AssistantScopeAsset & { type: AssistantAssetType }>) { return assets.find((item) => item.type === type && item.id === id)?.name ?? `${formatType(type)} ${id.slice(0, 8)}`; }
function actionLabel(value: string) { return ({ start_cleaning: "启动清洗", activate_cleaning_version: "切换清洗版本", rollback_cleaning_version: "回滚清洗版本", update_column_metadata: "更新字段元数据", save_relationship_plan: "保存数据关系", start_analysis: "启动分析", cancel_analysis: "取消分析", retry_analysis: "重试分析", rename_report: "重命名报告", create_semantic_draft: "创建语义草稿", update_semantic_draft: "更新语义草稿", publish_semantic_model: "发布语义模型", soft_delete_asset: "移入回收站", restore_asset: "恢复资产" } as Record<string, string>)[value] ?? value; }
