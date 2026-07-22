import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  FileImage,
  FileSpreadsheet,
  Loader2,
  MessageSquareText,
  Paperclip,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { apiDelete, apiFetch, apiGet, apiPatch, apiPost, apiPostForm } from "../../api-client";
import { AssistantAttachmentImage } from "./AssistantAttachmentImage";
import { AssistantControlPanel } from "./AssistantControlPanel";
import { AssistantConversationHeader } from "./AssistantConversationHeader";
import { AssistantEvidenceCards } from "./AssistantEvidenceCards";
import { AssistantImportPreview } from "./AssistantImportPreview";
import { AssistantWorkflowCard } from "./AssistantWorkflowCard";
import { ASSISTANT_FULL_CAPABILITIES } from "./types";
import type {
  AssistantAttachment,
  AssistantConversation,
  AssistantEvent,
  AssistantExecutionMode,
  AssistantImportBatch,
  AssistantMessage,
  AssistantRun,
  AssistantScopeAsset,
  AssistantScopeType,
} from "./types";

type Props = {
  datasets: AssistantScopeAsset[];
  datasetGroups: AssistantScopeAsset[];
  reports: AssistantScopeAsset[];
  onActiveRunsChange?: (count: number) => void;
  onAssetsChanged?: () => void;
  onOpenDataset: (id: string) => void;
  onOpenAnalysis: (id: string) => void;
  onOpenReport: (id: string) => void;
};

export function AssistantPage({ datasets, datasetGroups, reports, onActiveRunsChange, onAssetsChanged, onOpenDataset, onOpenAnalysis, onOpenReport }: Props) {
  const [conversations, setConversations] = useState<AssistantConversation[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [run, setRun] = useState<AssistantRun | null>(null);
  const [events, setEvents] = useState<AssistantEvent[]>([]);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<AssistantAttachment[]>([]);
  const [executionMode, setExecutionMode] = useState<AssistantExecutionMode>("ask");
  const [importBatch, setImportBatch] = useState<AssistantImportBatch | null>(null);
  const [sheetSelections, setSheetSelections] = useState<Record<string, string>>({});
  const [importing, setImporting] = useState(false);
  const [controlOpen, setControlOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historySearch, setHistorySearch] = useState("");
  const [historyCollapsed, setHistoryCollapsed] = useState(() => window.localStorage.getItem("datamind:assistant-history-collapsed") === "true");
  const [workbenchSummary, setWorkbenchSummary] = useState({ grants: 0, actions: 0, recycled: 0 });
  const endRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const currentIdRef = useRef<string | null>(null);

  const current = conversations.find((item) => item.conversation_id === currentId) ?? null;
  const active = !!run && ["queued", "running", "cancel_requested"].includes(run.status);
  const filteredConversations = conversations.filter((item) => item.title.toLocaleLowerCase().includes(historySearch.trim().toLocaleLowerCase()));
  const workbenchCount = workbenchSummary.grants + workbenchSummary.actions + workbenchSummary.recycled;

  const toggleHistoryCollapsed = () => {
    setHistoryCollapsed((value) => {
      const next = !value;
      window.localStorage.setItem("datamind:assistant-history-collapsed", String(next));
      return next;
    });
  };

  const refreshConversations = async (selectFirst = false) => {
    const payload = await apiGet<{ conversations: AssistantConversation[] }>("/assistant/conversations");
    setConversations(payload.conversations);
    const next = currentId && payload.conversations.some((item) => item.conversation_id === currentId)
      ? currentId
      : selectFirst ? payload.conversations[0]?.conversation_id ?? null : currentId;
    setCurrentId(next);
    const activeCount = payload.conversations.filter((item) => item.active_run_id).length;
    onActiveRunsChange?.(activeCount);
    return { conversations: payload.conversations, selected: next };
  };

  const refreshMessages = async (conversationId = currentId) => {
    if (!conversationId) {
      setMessages([]);
      return;
    }
    const payload = await apiGet<{ messages: AssistantMessage[] }>(`/assistant/conversations/${conversationId}/messages`);
    setMessages(payload.messages);
  };

  useEffect(() => {
    void (async () => {
      try {
        const state = await refreshConversations(true);
        if (!state.selected && state.conversations.length === 0) {
          const created = await apiPost<AssistantConversation>("/assistant/conversations", { scope_type: "auto" });
          setCurrentId(created.conversation_id);
          await refreshConversations();
        }
      } catch (cause) {
        setError(messageOf(cause));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    currentIdRef.current = currentId;
    setAttachments([]);
    setImportBatch(null);
    setSheetSelections({});
    setEvents([]);
    setRun(null);
    if (!currentId) return;
    void refreshMessages(currentId).catch((cause) => setError(messageOf(cause)));
    const conversation = conversations.find((item) => item.conversation_id === currentId);
    if (conversation?.active_run_id) void resumeRun(conversation.active_run_id);
  }, [currentId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, events, run?.status]);

  const createConversation = async () => {
    try {
      const created = await apiPost<AssistantConversation>("/assistant/conversations", { scope_type: "auto" });
      setConversations((items) => [created, ...items]);
      setCurrentId(created.conversation_id);
      setHistoryOpen(false);
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const renameConversation = async (item: AssistantConversation) => {
    const title = window.prompt("对话名称", item.title)?.trim();
    if (!title || title === item.title) return;
    try {
      const updated = await apiPatch<AssistantConversation>(`/assistant/conversations/${item.conversation_id}`, { title });
      setConversations((items) => items.map((value) => value.conversation_id === updated.conversation_id ? updated : value));
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const deleteConversation = async (item: AssistantConversation) => {
    if (!window.confirm(`删除对话“${item.title}”？`)) return;
    try {
      await apiDelete(`/assistant/conversations/${item.conversation_id}`);
      const remaining = conversations.filter((value) => value.conversation_id !== item.conversation_id);
      setConversations(remaining);
      if (currentId === item.conversation_id) setCurrentId(remaining[0]?.conversation_id ?? null);
      if (!remaining.length) await createConversation();
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const updateScope = async (value: string) => {
    if (!current) return;
    const [scopeType, scopeId] = value.split(":", 2) as [AssistantScopeType, string | undefined];
    try {
      const updated = await apiPatch<AssistantConversation>(`/assistant/conversations/${current.conversation_id}`, { scope_type: scopeType, scope_id: scopeId || null });
      if (scopeType !== "auto" && scopeId) {
        try {
          await apiPost("/assistant/permission-grants", {
            asset_type: scopeType,
            asset_id: scopeId,
            capabilities: ASSISTANT_FULL_CAPABILITIES,
          });
        } catch (cause) {
          setError(`范围已切换，但自动授权失败：${messageOf(cause)}`);
        }
      }
      setConversations((items) => items.map((item) => item.conversation_id === updated.conversation_id ? updated : item));
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const uploadFiles = async (files: File[]) => {
    if (!currentId) return;
    if (files.length > 20) {
      setError("一次最多选择 20 个文件。");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const uploaded: AssistantAttachment[] = [];
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        uploaded.push(await apiPostForm<AssistantAttachment>(`/assistant/conversations/${currentId}/attachments`, form, 240000));
      }
      setAttachments((items) => [...items, ...uploaded]);
      setImportBatch(null);
      setSheetSelections({});
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const previewImport = async () => {
    if (!currentId) return;
    const ids = attachments.filter((item) => item.attachment_kind === "data_file").map((item) => item.attachment_id);
    if (!ids.length) return;
    setImporting(true);
    setError(null);
    try {
      setImportBatch(await apiPost<AssistantImportBatch>("/assistant/import-batches/preview", { conversation_id: currentId, attachment_ids: ids }, 240000));
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setImporting(false);
    }
  };

  const commitImport = async (allowPartial = false) => {
    if (!importBatch) return;
    setImporting(true);
    setError(null);
    try {
      const committed = await apiPost<AssistantImportBatch>(`/assistant/import-batches/${importBatch.batch_id}/commit`, { allow_partial: allowPartial, sheet_selections: sheetSelections }, 240000);
      setImportBatch(committed);
      setAttachments((items) => items.filter((item) => item.attachment_kind !== "data_file"));
      setExecutionMode("execute");
      onAssetsChanged?.();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setImporting(false);
    }
  };

  const send = async () => {
    if (!currentId || !draft.trim() || active || uploading) return;
    const content = draft.trim();
    setDraft("");
    setError(null);
    try {
      const created = await apiPost<AssistantRun>(`/assistant/conversations/${currentId}/messages`, { content, attachment_ids: attachments.filter((item) => item.attachment_kind === "image").map((item) => item.attachment_id), execution_mode: executionMode }, 60000);
      setAttachments([]);
      setRun(created);
      setEvents([]);
      await refreshMessages(currentId);
      await streamRun(created);
    } catch (cause) {
      setDraft(content);
      setError(messageOf(cause));
    }
  };

  const resumeRun = async (runId: string) => {
    try {
      const value = await apiGet<AssistantRun>(`/assistant/runs/${runId}`);
      setRun(value);
      if (["queued", "running", "cancel_requested"].includes(value.status)) await streamRun(value);
      else await refreshMessages(value.conversation_id);
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const streamRun = (initial: AssistantRun) => new Promise<void>((resolve) => {
    let lastSequence = initial.last_event_sequence ?? 0;
    let settled = false;
    const controller = new AbortController();
    const finish = async () => {
      if (settled) return;
      settled = true;
      controller.abort();
      try {
        const latest = await apiGet<AssistantRun>(`/assistant/runs/${initial.run_id}`);
        if (currentIdRef.current === latest.conversation_id) {
          setRun(latest);
          await refreshMessages(latest.conversation_id);
        }
        await refreshConversations();
      } catch (cause) {
        setError(messageOf(cause));
      }
      resolve();
    };
    const handleEvent = (raw: string) => {
      try {
        const event = JSON.parse(raw) as AssistantEvent;
        lastSequence = Math.max(lastSequence, event.sequence);
        if (currentIdRef.current === initial.conversation_id) {
          setEvents((items) => [...items.filter((item) => item.sequence !== event.sequence), event].sort((a, b) => a.sequence - b.sequence));
        }
        if (event.event_type === "message.delta") {
          const delta = String(event.payload.delta ?? "");
          if (currentIdRef.current === initial.conversation_id) {
            setMessages((items) => items.map((item) => item.message_id === initial.assistant_message_id ? { ...item, content: item.content + delta, status: "streaming" } : item));
          }
        }
      } catch {
        // Malformed events are ignored; the final message is reloaded from the server.
      }
    };
    void consumeAssistantEvents(initial.run_id, lastSequence, controller.signal, handleEvent)
      .then(() => void finish())
      .catch(() => {
        controller.abort();
        void pollRun(initial.run_id).then(finish);
      });
  });

  const pollRun = async (runId: string) => {
    for (let attempt = 0; attempt < 300; attempt += 1) {
      const latest = await apiGet<AssistantRun>(`/assistant/runs/${runId}`);
      if (currentIdRef.current === latest.conversation_id) setRun(latest);
      if (!["queued", "running", "cancel_requested"].includes(latest.status)) return;
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
  };

  const confirm = async (accepted: boolean) => {
    if (!run) return;
    try {
      const next = await apiPost<AssistantRun>(`/assistant/runs/${run.run_id}/confirm`, { accepted });
      setRun(next);
      if (accepted) await streamRun(next);
      else await refreshMessages(next.conversation_id);
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const cancel = async () => {
    if (!run) return;
    try {
      setRun(await apiPost<AssistantRun>(`/assistant/runs/${run.run_id}/cancel`, {}));
    } catch (cause) {
      setError(messageOf(cause));
    }
  };

  const scopeValue = current ? `${current.scope_type}${current.scope_id ? `:${current.scope_id}` : ""}` : "auto";

  return (
    <section className="assistant-shell">
      <div className={`assistant-workspace ${controlOpen ? "has-control" : ""} ${historyCollapsed ? "history-collapsed" : ""}`}>
        {historyOpen && <button type="button" className="assistant-history-scrim" aria-label="关闭消息记录" onClick={() => setHistoryOpen(false)} />}
        <aside className={`assistant-history ${historyOpen ? "open" : ""} ${historyCollapsed ? "collapsed" : ""}`}>
          <div className="assistant-history-heading">
            <strong>消息记录</strong>
            <div>
              <button type="button" className="icon-button" title="新建对话" onClick={() => void createConversation()}><Plus size={18} /></button>
              <button type="button" className="icon-button assistant-history-toggle" title={historyCollapsed ? "展开消息记录" : "收起消息记录"} onClick={toggleHistoryCollapsed}>{historyCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}</button>
            </div>
          </div>
          <label className="assistant-history-search"><Search size={14} /><input aria-label="搜索对话" value={historySearch} onChange={(event) => setHistorySearch(event.target.value)} placeholder="搜索对话" /></label>
          <div className="assistant-conversation-list">
            {filteredConversations.map((item) => (
              <div key={item.conversation_id} className={`assistant-conversation ${item.conversation_id === currentId ? "active" : ""}`}>
                <button type="button" title={historyCollapsed ? item.title : undefined} onClick={() => { setCurrentId(item.conversation_id); setHistoryOpen(false); }}>
                  <MessageSquareText size={16} />
                  <span><b>{item.title}</b><small>{formatTime(item.last_message_at ?? item.created_at)}</small></span>
                  {item.active_run_id && <Loader2 size={14} className="animate-spin text-emerald-600" />}
                </button>
                <div className="assistant-conversation-actions">
                  <button type="button" title="重命名" onClick={() => void renameConversation(item)}><Pencil size={13} /></button>
                  <button type="button" title="删除" onClick={() => void deleteConversation(item)}><Trash2 size={13} /></button>
                </div>
              </div>
            ))}
          </div>
          <button type="button" className="assistant-history-close md:hidden" onClick={() => setHistoryOpen(false)}><X size={16} /> 关闭</button>
        </aside>

        <div className="assistant-thread">
          <AssistantConversationHeader executionMode={executionMode} scopeValue={scopeValue} datasets={datasets} datasetGroups={datasetGroups} reports={reports} active={active} workbenchCount={workbenchCount} onScopeChange={(value) => void updateScope(value)} onOpenHistory={() => { setControlOpen(false); setHistoryOpen(true); }} onOpenWorkbench={() => { setHistoryOpen(false); setControlOpen(true); }} />

          <div className="assistant-messages">
            {loading && <div className="assistant-empty"><Loader2 className="animate-spin" /><p>正在读取消息记录...</p></div>}
            {!loading && !messages.length && (
              <div className="assistant-empty">
                <span><Sparkles size={26} /></span>
                <h3>从你的数据开始</h3>
                <p>Kimi 会优先检索已经完成的分析和报告，需要新结论时会调用 DataMind Workflow。</p>
                <div className="assistant-suggestions">
                  {["概括最近一份分析报告", "哪些结论有数据质量风险？", "基于现有数据建议下一步分析"].map((value) => <button key={value} type="button" onClick={() => setDraft(value)}>{value}<ArrowRight size={14} /></button>)}
                </div>
              </div>
            )}
            {messages.map((message) => (
              <article key={message.message_id} className={`assistant-message ${message.role}`}>
                <div className="assistant-message-avatar">{message.role === "user" ? "你" : <Sparkles size={16} />}</div>
                <div className="assistant-message-body">
                  <div className="assistant-message-meta"><b>{message.role === "user" ? "你" : "Kimi"}</b><span>{formatTime(message.created_at)}</span>{message.model && <span>{message.model}</span>}</div>
                  {message.attachments.length > 0 && <div className="assistant-message-images">{message.attachments.map((item) => item.attachment_kind === "image" ? <AssistantAttachmentImage key={item.attachment_id} attachment={item} /> : <div className="assistant-message-file" key={item.attachment_id}><FileSpreadsheet size={18} /><span><b>{item.file_name}</b><small>{item.import_status ?? "数据文件"}</small></span></div>)}</div>}
                  <MarkdownText text={message.content || (message.status === "pending" ? "正在准备回答..." : "")} />
                  <AssistantEvidenceCards citations={message.citations} onOpenDataset={onOpenDataset} onOpenAnalysis={onOpenAnalysis} onOpenReport={onOpenReport} />
                </div>
              </article>
            ))}

            {run && (active || run.status === "awaiting_confirmation" || run.status === "failed") && (
              <AssistantWorkflowCard run={run} events={events} active={active} onCancel={() => void cancel()} onConfirm={(accepted) => void confirm(accepted)} friendlyError={friendlyRunError} eventLabel={toolLabel} />
            )}
            <div ref={endRef} />
          </div>

          <div className="assistant-composer-wrap">
            {error && <div className="assistant-error"><span>{error}</span><button type="button" onClick={() => setError(null)}><X size={14} /></button></div>}
            {attachments.length > 0 && <div className="assistant-attachment-strip">{attachments.map((item) => <div key={item.attachment_id}>{item.attachment_kind === "data_file" ? <FileSpreadsheet size={15} /> : <FileImage size={15} />}<span>{item.file_name}<small>{formatBytes(item.size_bytes)}</small></span><button type="button" onClick={() => { setAttachments((values) => values.filter((value) => value.attachment_id !== item.attachment_id)); setImportBatch(null); }}><X size={13} /></button></div>)}</div>}
            {attachments.some((item) => item.attachment_kind === "data_file") && <AssistantImportPreview batch={importBatch} importing={importing} sheetSelections={sheetSelections} onSheetChange={(attachmentId, sheet) => setSheetSelections((values) => ({ ...values, [attachmentId]: sheet }))} onPreview={() => void previewImport()} onCommit={(allowPartial) => void commitImport(allowPartial)} />}
            <div className="assistant-composer">
              <div className="assistant-mode-switch" aria-label="Kimi 模式"><button type="button" className={executionMode === "ask" ? "active" : ""} disabled={active} onClick={() => setExecutionMode("ask")}>问答</button><button type="button" className={executionMode === "execute" ? "active" : ""} disabled={active} onClick={() => setExecutionMode("execute")}>执行任务</button><span>{executionMode === "ask" ? "只读，不会修改数据" : "仅在已授权资产内执行"}</span></div>
              <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onPaste={(event) => { const file = Array.from(event.clipboardData.files).find((item) => item.type.startsWith("image/")); if (file) { event.preventDefault(); void uploadFiles([file]); } }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} placeholder={executionMode === "execute" ? "描述目标，例如：清洗这份数据并分析影响销售额的因素..." : "向 Kimi 询问你的数据、分析结果或报告..."} rows={2} disabled={active} />
              <div className="assistant-composer-actions">
                <div><button type="button" className="icon-button" title="上传图片或数据文件" disabled={uploading || active} onClick={() => fileRef.current?.click()}>{uploading ? <Loader2 className="animate-spin" size={18} /> : <Paperclip size={18} />}</button><input ref={fileRef} type="file" multiple accept="image/jpeg,image/png,image/webp,.csv,.xlsx,.json,.txt" hidden onChange={(event) => { const files = Array.from(event.target.files ?? []); if (files.length) void uploadFiles(files); }} /><span>图片最大 5MB · 数据文件最大 200MB · 每批最多 20 个</span></div>
                <button type="button" className="assistant-send" disabled={!draft.trim() || active || uploading || attachments.some((item) => item.attachment_kind === "data_file")} onClick={() => void send()}>{active ? <Loader2 className="animate-spin" size={17} /> : <Send size={17} />}<span>{active ? "运行中" : executionMode === "execute" ? "执行" : "发送"}</span></button>
              </div>
            </div>
          </div>
        </div>
        <AssistantControlPanel
          open={controlOpen}
          onClose={() => setControlOpen(false)}
          onSummaryChange={setWorkbenchSummary}
          scopeType={current?.scope_type ?? "auto"}
          scopeId={current?.scope_id ?? null}
          datasets={datasets}
          datasetGroups={datasetGroups}
          reports={reports}
        />
      </div>
    </section>
  );
}

async function consumeAssistantEvents(runId: string, afterSequence: number, signal: AbortSignal, onEvent: (data: string) => void) {
  const response = await apiFetch(`/assistant/runs/${runId}/events?after_sequence=${afterSequence}`, {
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (!response.ok || !response.body) throw new Error(`Assistant event stream unavailable: ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const eventName = frame.split(/\r?\n/).find((line) => line.startsWith("event:"))?.slice(6).trim();
      const data = frame.split(/\r?\n/).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
      if (eventName === "assistant" && data) onEvent(data);
      if (eventName === "end") return;
    }
  }
}

function friendlyRunError(error: string) {
  if (error.includes("Kimi API error 401")) return "Kimi API Key 无效或已失效，请更新密钥后重试。";
  if (error.includes("Kimi API error 429")) return "Kimi 当前请求较多或额度不足，请稍后重试。";
  if (error.includes("invalid temperature")) return "Kimi 模型参数不兼容，请刷新页面后重新发送。";
  return "Kimi 本次执行失败。请重试；若持续失败，可查看后端运行日志。";
}

function MarkdownText({ text }: { text: string }) {
  const blocks = useMemo(() => text.split(/\n{2,}/), [text]);
  return <div className="assistant-markdown">{blocks.map((block, index) => {
    if (block.startsWith("```")) return <pre key={index}><code>{block.replace(/^```\w*\n?/, "").replace(/```$/, "")}</code></pre>;
    if (/^#{1,3}\s/.test(block)) return <h4 key={index}>{block.replace(/^#{1,3}\s*/, "")}</h4>;
    if (block.split("\n").every((line) => /^[-*]\s/.test(line))) return <ul key={index}>{block.split("\n").map((line) => <li key={line}>{inlineText(line.replace(/^[-*]\s*/, ""))}</li>)}</ul>;
    return <p key={index}>{block.split("\n").map((line, lineIndex) => <React.Fragment key={`${lineIndex}-${line}`}>{lineIndex > 0 && <br />}{inlineText(line)}</React.Fragment>)}</p>;
  })}</div>;
}

function inlineText(value: string) {
  return value.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => part.startsWith("**") ? <strong key={index}>{part.slice(2, -2)}</strong> : part.startsWith("`") ? <code key={index}>{part.slice(1, -1)}</code> : part);
}

function toolLabel(event: AssistantEvent) {
  const labels: Record<string, string> = { search_datamind_assets: "检索 DataMind 资产", get_dataset_context: "读取数据集结构", get_analysis_result: "读取分析结果", get_report: "读取报告", preview_analysis_plan: "规划分析", start_analysis: "运行 DataMind Workflow", get_analysis_status: "检查分析状态", start_cleaning: "运行自主清洗", get_cleaning_status: "检查清洗状态", activate_cleaning_version: "激活清洗版本", rollback_cleaning_version: "回滚清洗版本", update_column_metadata: "更新字段元数据", suggest_relationships: "生成关系建议", save_relationship_plan: "保存关系计划", cancel_analysis: "取消分析", retry_analysis: "重试分析", rename_report: "重命名报告", create_semantic_draft: "创建语义草稿", update_semantic_draft: "更新语义草稿", validate_semantic_model: "校验语义模型", publish_semantic_model: "发布语义模型", soft_delete_asset: "移入回收站", restore_asset: "恢复资产" };
  const eventLabels: Record<string, string> = { "retrieval.completed": "自动检索", "message.completed": "生成回答", "permission.checked": "权限校验", "action.planned": "准备执行", "action.completed": "操作完成", "action.rolled_back": "操作已撤销", "import.progress": "数据包导入", "asset.recycled": "资产回收" };
  return labels[event.tool_name ?? ""] ?? eventLabels[event.event_type] ?? "Kimi 工具";
}

function formatTime(value?: string | null) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function formatBytes(value: number) {
  if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(value / 1024))} KB`;
}

function messageOf(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
