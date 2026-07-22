import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2, Plus, RotateCcw } from "lucide-react";

export type CleaningDiffSummary = {
  raw_row_count: number;
  previous_row_count: number;
  current_row_count: number;
  added_rows: number;
  removed_rows: number;
  changed_rows: number;
  added_columns: string[];
  removed_columns: string[];
  changed_cells: number;
  raw_missing_count: number;
  previous_missing_count: number;
  current_missing_count: number;
  sample_diffs: Record<string, unknown>[];
};

export type CleaningRunDetail = {
  id: string;
  dataset_id: string;
  version: number;
  is_active: boolean;
  provider: string;
  model: string;
  prompt: string;
  result_markdown: string;
  cleaned_dataset: Record<string, unknown>;
  raw_summary: Record<string, unknown>;
  previous_summary: Record<string, unknown>;
  current_summary: Record<string, unknown>;
  diff_summary: CleaningDiffSummary;
  created_at?: string | null;
};

export type CleaningRule = {
  rule_type: "fill_missing" | "drop_duplicates" | "rename_column" | "convert_type" | "trim_text" | "drop_column" | "filter_rows";
  column?: string | null;
  value?: unknown;
  new_name?: string | null;
  target_type?: "text" | "number" | "integer" | "float" | "date" | "boolean" | null;
  strategy?: "empty_string" | "zero" | "value" | "drop_row" | null;
  operator?: "equals" | "not_equals" | "contains" | "not_contains" | "blank" | "not_blank" | "gt" | "gte" | "lt" | "lte" | null;
  mode?: "keep" | "delete" | null;
  enabled: boolean;
};

export type CleaningRulePreviewResponse = {
  dataset_id: string;
  preview_records: Record<string, unknown>[];
  diff_summary: CleaningDiffSummary;
  validation_issues: string[];
  applied_rules: CleaningRule[];
};

type DatasetColumnProfile = {
  name: string;
  dtype: string;
  is_numeric: boolean;
  mean?: number | null;
};

type DatasetProfile = {
  columns: DatasetColumnProfile[];
};

type DatasetColumnMetadata = {
  column_name: string;
  inferred_type: string;
  override_type?: string | null;
  description: string;
  role: "dimension" | "metric" | "id" | "text" | "date" | "ignore";
  created_at?: string | null;
  updated_at?: string | null;
};

export function CleaningVersionsPanel({ runs, onActivate }: { runs: CleaningRunDetail[]; onActivate: (runId: string) => Promise<void> }) {
  const [busyRunId, setBusyRunId] = useState<string | null>(null);
  if (!runs.length) return <Panel title="清洗版本"><Notice>当前数据集还没有清洗版本。</Notice></Panel>;
  const activate = async (run: CleaningRunDetail) => {
    setBusyRunId(run.id);
    try { await onActivate(run.id); } finally { setBusyRunId(null); }
  };
  return (
    <Panel title="清洗版本">
      <div className="grid gap-3">
        {runs.map((run) => {
          const diff = run.diff_summary;
          return <details key={run.id} className="rounded-lg border border-slate-200 bg-white p-4" open={run.is_active}>
            <summary className="cursor-pointer font-black marker:text-emerald-700">v{run.version} · {run.is_active ? "当前版本" : "历史版本"} · {formatTime(run.created_at)}</summary>
            <div className="mt-4 grid gap-4">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4"><Metric label="行数变化" value={`${diff.previous_row_count} → ${diff.current_row_count}`} /><Metric label="新增/删除行" value={`+${diff.added_rows} / -${diff.removed_rows}`} /><Metric label="修改行" value={String(diff.changed_rows)} /><Metric label="空值变化" value={`${diff.previous_missing_count} → ${diff.current_missing_count}`} /></div>
              <div className="grid gap-2 text-sm text-slate-600 md:grid-cols-2"><FieldChange label="新增字段" values={diff.added_columns} /><FieldChange label="删除字段" values={diff.removed_columns} /></div>
              <PreviewTable rows={diff.sample_diffs} empty="没有单元格级样本差异。" />
              <pre className="prose-block">{run.result_markdown}</pre>
              {!run.is_active && <button type="button" className="small-button w-fit" disabled={busyRunId === run.id} onClick={() => void activate(run)}>{busyRunId === run.id ? <Loader2 className="animate-spin" size={15} /> : <RotateCcw size={15} />}设为当前版本</button>}
            </div>
          </details>;
        })}
      </div>
    </Panel>
  );
}

export function CleaningRuleEditor({ columns, onPreview, onApply }: { columns: string[]; onPreview: (rules: CleaningRule[]) => Promise<CleaningRulePreviewResponse>; onApply: (rules: CleaningRule[]) => Promise<void> }) {
  const [rules, setRules] = useState<CleaningRule[]>([{ rule_type: "trim_text", column: columns[0] ?? "", enabled: true }]);
  const [preview, setPreview] = useState<CleaningRulePreviewResponse | null>(null);
  const [riskConfirmed, setRiskConfirmed] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const riskReasons = useMemo(() => preview ? cleaningRuleRiskReasons(preview, rules) : [], [preview, rules]);
  useEffect(() => {
    if (!rules.length && columns.length) setRules([{ rule_type: "trim_text", column: columns[0], enabled: true }]);
  }, [columns.join("|")]);
  const invalidate = () => { setPreview(null); setRiskConfirmed(false); };
  const updateRule = (index: number, patch: Partial<CleaningRule>) => { setRules((current) => current.map((rule, ruleIndex) => ruleIndex === index ? { ...rule, ...patch } : rule)); invalidate(); };
  const runPreview = async () => {
    setBusy(true); setMessage(null); setRiskConfirmed(false);
    try { setPreview(await onPreview(rules)); } catch (error) { setMessage(readableError(error)); } finally { setBusy(false); }
  };
  const runApply = async () => {
    if (!preview) return setMessage("请先预览 diff，再确认应用当前规则。");
    if (preview.validation_issues.length) return setMessage("当前预览存在校验问题，不能创建清洗版本。");
    if (riskReasons.length && !riskConfirmed) return setMessage("这是高影响清洗，请先确认影响范围。");
    setBusy(true); setMessage(null);
    try { await onApply(rules); setMessage("清洗规则已应用为新的清洗版本。"); setPreview(null); setRiskConfirmed(false); } catch (error) { setMessage(readableError(error)); } finally { setBusy(false); }
  };
  return (
    <Panel title="清洗规则编辑器" action={<button type="button" className="small-button" onClick={() => { setRules((current) => [...current, { rule_type: "trim_text", column: columns[0] ?? "", enabled: true }]); invalidate(); }}><Plus size={15} />添加规则</button>}>
      <div className="grid gap-3">
        {rules.map((rule, index) => <RuleRow key={index} rule={rule} columns={columns} onChange={(patch) => updateRule(index, patch)} onRemove={() => { setRules((current) => current.filter((_, item) => item !== index)); invalidate(); }} />)}
      </div>
      <div className="mt-4 flex flex-wrap gap-3"><button type="button" className="small-button" disabled={busy || !rules.length} onClick={() => void runPreview()}>{busy ? <Loader2 className="animate-spin" size={15} /> : null}预览 diff</button><button type="button" className="small-button" disabled={busy || !rules.length || !preview || !!preview.validation_issues.length || (!!riskReasons.length && !riskConfirmed)} onClick={() => void runApply()}>{riskReasons.length ? "确认并创建版本" : "应用为新版本"}</button></div>
      {message && <Notice tone={/失败|错误|不能|请先/i.test(message) ? "error" : "info"}>{message}</Notice>}
      {preview && <div className="mt-5 grid gap-4">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4"><Metric label="行数变化" value={`${preview.diff_summary.previous_row_count} → ${preview.diff_summary.current_row_count}`} /><Metric label="修改行" value={String(preview.diff_summary.changed_rows)} /><Metric label="修改单元格" value={String(preview.diff_summary.changed_cells)} /><Metric label="空值变化" value={`${preview.diff_summary.previous_missing_count} → ${preview.diff_summary.current_missing_count}`} /></div>
        {!!preview.validation_issues.length && <Notice tone="error">{preview.validation_issues.join("；")}</Notice>}
        {!!riskReasons.length && <RiskConfirmation reasons={riskReasons} checked={riskConfirmed} onChange={setRiskConfirmed} />}
        <PreviewTable rows={preview.preview_records} empty="没有预览记录。" />
        <PreviewTable rows={preview.diff_summary.sample_diffs} empty="没有样本 diff。" />
      </div>}
    </Panel>
  );
}

export function ColumnMetadataEditor({
  profile,
  metadata,
  onUpdateColumn,
}: {
  profile: DatasetProfile | null;
  metadata: DatasetColumnMetadata[];
  onUpdateColumn: (
    columnName: string,
    payload: Partial<DatasetColumnMetadata>,
  ) => Promise<void>;
}) {
  const metadataByColumn = new Map(metadata.map((item) => [item.column_name, item]));
  const [savingColumn, setSavingColumn] = useState<string | null>(null);
  if (!profile?.columns.length) {
    return <Panel title="字段类型与描述"><Notice>暂无字段可编辑。</Notice></Panel>;
  }
  const update = async (
    columnName: string,
    payload: Partial<DatasetColumnMetadata>,
  ) => {
    setSavingColumn(columnName);
    try {
      await onUpdateColumn(columnName, payload);
    } finally {
      setSavingColumn(null);
    }
  };
  return (
    <Panel title="字段类型与描述">
      <div className="overflow-auto rounded-xl border border-line">
        <table className="w-full min-w-[900px] border-collapse text-left text-sm">
          <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              {["字段", "推断类型", "覆盖类型", "角色", "描述", "操作"].map((label) => (
                <th key={label} className="px-3 py-3">{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profile.columns.map((column) => {
              const item = metadataByColumn.get(column.name);
              return (
                <ColumnMetadataRow
                  key={column.name}
                  column={column}
                  metadata={item}
                  inferredType={item?.inferred_type || inferredTypeFromProfile(column)}
                  saving={savingColumn === column.name}
                  onSave={(payload) => update(column.name, payload)}
                />
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function ColumnMetadataRow({
  column,
  metadata,
  inferredType,
  saving,
  onSave,
}: {
  column: DatasetColumnProfile;
  metadata?: DatasetColumnMetadata;
  inferredType: string;
  saving: boolean;
  onSave: (payload: Partial<DatasetColumnMetadata>) => Promise<void>;
}) {
  const [overrideType, setOverrideType] = useState(metadata?.override_type ?? "");
  const [role, setRole] = useState<DatasetColumnMetadata["role"]>(
    metadata?.role ?? (column.is_numeric ? "metric" : "dimension"),
  );
  const [description, setDescription] = useState(metadata?.description ?? "");
  useEffect(() => {
    setOverrideType(metadata?.override_type ?? "");
    setRole(metadata?.role ?? (column.is_numeric ? "metric" : "dimension"));
    setDescription(metadata?.description ?? "");
  }, [metadata?.override_type, metadata?.role, metadata?.description, column.is_numeric]);
  return (
    <tr className="border-b border-line bg-white align-top">
      <td className="px-3 py-3 font-black">{column.name}</td>
      <td className="px-3 py-3">{inferredType}</td>
      <td className="px-3 py-3">
        <select value={overrideType} onChange={(event) => setOverrideType(event.target.value)} className="input py-2">
          <option value="">自动</option>
          {["text", "number", "integer", "float", "date", "boolean"].map((type) => <option key={type} value={type}>{type}</option>)}
        </select>
      </td>
      <td className="px-3 py-3">
        <select value={role} onChange={(event) => setRole(event.target.value as DatasetColumnMetadata["role"])} className="input py-2">
          {["dimension", "metric", "id", "text", "date", "ignore"].map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </td>
      <td className="px-3 py-3"><input value={description} onChange={(event) => setDescription(event.target.value)} className="input py-2" /></td>
      <td className="px-3 py-3">
        <div className="flex gap-2">
          <button type="button" className="small-button" disabled={saving} onClick={() => void onSave({ inferred_type: inferredType, override_type: overrideType || null, role, description })}>
            {saving ? <Loader2 className="animate-spin" size={14} /> : null}
            保存
          </button>
          <button type="button" className="small-button bg-slate-700 hover:bg-slate-800" disabled={saving} onClick={() => void onSave({ inferred_type: inferredType, override_type: null, role: column.is_numeric ? "metric" : "dimension", description: "" })}>
            恢复
          </button>
        </div>
      </td>
    </tr>
  );
}

function RuleRow({ rule, columns, onChange, onRemove }: { rule: CleaningRule; columns: string[]; onChange: (patch: Partial<CleaningRule>) => void; onRemove: () => void }) {
  return <div className="rounded-lg border border-slate-200 bg-white p-4">
    <div className="grid gap-3 md:grid-cols-[1.1fr_1.1fr_1fr_1fr_1fr_auto]">
      <select value={rule.rule_type} onChange={(event) => onChange({ rule_type: event.target.value as CleaningRule["rule_type"] })} className="input py-2">{RULE_TYPES.map((type) => <option key={type} value={type}>{RULE_LABELS[type]}</option>)}</select>
      <select value={rule.column ?? ""} onChange={(event) => onChange({ column: event.target.value })} className="input py-2" disabled={rule.rule_type === "drop_duplicates"}><option value="">选择字段</option>{columns.map((column) => <option key={column}>{column}</option>)}</select>
      <input value={String(rule.value ?? "")} onChange={(event) => onChange({ value: event.target.value })} className="input py-2" placeholder="值/条件值" />
      <input value={rule.new_name ?? ""} onChange={(event) => onChange({ new_name: event.target.value })} className="input py-2" placeholder="新字段名" />
      <select value={rule.target_type ?? ""} onChange={(event) => onChange({ target_type: (event.target.value || null) as CleaningRule["target_type"] })} className="input py-2"><option value="">目标类型</option>{["text", "number", "integer", "float", "date", "boolean"].map((type) => <option key={type}>{type}</option>)}</select>
      <button type="button" className="small-button bg-slate-700 hover:bg-slate-800" onClick={onRemove}>删除</button>
    </div>
    <div className="mt-3 grid gap-3 md:grid-cols-3">
      <select value={rule.strategy ?? ""} onChange={(event) => onChange({ strategy: (event.target.value || null) as CleaningRule["strategy"] })} className="input py-2"><option value="">空值策略</option><option value="value">填入指定值</option><option value="empty_string">填空字符串</option><option value="zero">填 0</option><option value="drop_row">删除空值行</option></select>
      <select value={rule.operator ?? ""} onChange={(event) => onChange({ operator: (event.target.value || null) as CleaningRule["operator"] })} className="input py-2"><option value="">过滤条件</option>{["equals", "not_equals", "contains", "not_contains", "blank", "not_blank", "gt", "gte", "lt", "lte"].map((operator) => <option key={operator}>{operator}</option>)}</select>
      <select value={rule.mode ?? ""} onChange={(event) => onChange({ mode: (event.target.value || null) as CleaningRule["mode"] })} className="input py-2"><option value="">过滤模式</option><option value="keep">保留匹配行</option><option value="delete">删除匹配行</option></select>
    </div>
  </div>;
}

function RiskConfirmation({ reasons, checked, onChange }: { reasons: string[]; checked: boolean; onChange: (value: boolean) => void }) {
  return <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-amber-950"><div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 shrink-0" size={18} /><div><b className="text-sm">需要确认的高影响操作</b><ul className="mt-2 grid gap-1 text-sm font-semibold">{reasons.map((reason) => <li key={reason}>• {reason}</li>)}</ul><label className="mt-3 flex cursor-pointer items-center gap-2 text-sm font-black"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />我已核对行数、字段和样本 diff，确认创建可回滚的新版本</label></div></div></div>;
}

function cleaningRuleRiskReasons(preview: CleaningRulePreviewResponse, rules: CleaningRule[]) {
  const diff = preview.diff_summary;
  const reasons: string[] = [];
  const removedRatio = diff.previous_row_count > 0 ? diff.removed_rows / diff.previous_row_count : 0;
  if (diff.current_row_count === 0 && diff.previous_row_count > 0) reasons.push("规则会删除全部数据行。");
  if (removedRatio >= 0.1) reasons.push(`将删除 ${diff.removed_rows} 行，占当前数据的 ${(removedRatio * 100).toFixed(1)}%。`);
  if (diff.removed_columns.length) reasons.push(`将删除字段：${diff.removed_columns.join("、")}。`);
  if (diff.current_missing_count > diff.previous_missing_count) reasons.push(`缺失值将从 ${diff.previous_missing_count} 增加到 ${diff.current_missing_count}。`);
  const converted = rules.filter((rule) => rule.enabled && rule.rule_type === "convert_type").map((rule) => rule.column).filter(Boolean);
  if (converted.length) reasons.push(`字段类型将被转换：${converted.join("、")}；无法转换的值可能变为空值。`);
  if (rules.some((rule) => rule.enabled && rule.rule_type === "filter_rows")) reasons.push("包含行过滤规则，请确认筛选方向和条件值符合业务口径。");
  return Array.from(new Set(reasons));
}

function inferredTypeFromProfile(column: DatasetColumnProfile) {
  if (column.is_numeric) return Number.isInteger(column.mean ?? 0) ? "number" : "float";
  const lowered = column.name.toLowerCase();
  if (lowered.includes("date") || lowered.includes("time") || lowered.includes("日期") || lowered.includes("时间")) {
    return "date";
  }
  return column.dtype || "text";
}

const RULE_TYPES = ["fill_missing", "drop_duplicates", "rename_column", "convert_type", "trim_text", "drop_column", "filter_rows"] as const;
const RULE_LABELS: Record<(typeof RULE_TYPES)[number], string> = { fill_missing: "空值处理", drop_duplicates: "去重", rename_column: "列重命名", convert_type: "类型转换", trim_text: "文本 trim", drop_column: "删除列", filter_rows: "行过滤" };
function Panel({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) { return <section className="surface-card"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><h4 className="section-heading mb-0">{title}</h4>{action}</div>{children}</section>; }
function Metric({ label, value }: { label: string; value: string }) { return <div className="rounded-lg border border-line bg-slate-50 px-3 py-2"><p className="text-xs font-bold text-slate-500">{label}</p><p className="text-lg font-black text-slate-950">{value}</p></div>; }
function FieldChange({ label, values }: { label: string; values: string[] }) { return <div className="rounded-lg bg-slate-50 p-3"><b className="block text-slate-900">{label}</b>{values.length ? values.join(", ") : "无"}</div>; }
function Notice({ children, tone = "info" }: { children: React.ReactNode; tone?: "info" | "error" }) { return <div role={tone === "error" ? "alert" : "status"} className={`mt-4 rounded-lg border px-4 py-3 text-sm font-semibold ${tone === "error" ? "border-rose-200 bg-rose-50 text-rose-950" : "border-sky-200 bg-sky-50 text-slate-900"}`}>{children}</div>; }
function PreviewTable({ rows, empty }: { rows: Record<string, unknown>[]; empty: string }) { if (!rows.length) return <Notice>{empty}</Notice>; const columns = Object.keys(rows[0]).slice(0, 12); return <div className="overflow-auto rounded-lg border border-slate-200"><table className="w-full min-w-[620px] border-collapse text-left text-sm"><thead className="bg-slate-50 text-xs font-black text-slate-500"><tr>{columns.map((column) => <th key={column} className="px-3 py-2">{column}</th>)}</tr></thead><tbody>{rows.slice(0, 20).map((row, index) => <tr key={index} className="border-t border-slate-200 bg-white">{columns.map((column) => <td key={column} className="max-w-64 truncate px-3 py-2">{displayValue(row[column])}</td>)}</tr>)}</tbody></table></div>; }
function displayValue(value: unknown) { if (value === null || value === undefined) return "-"; return typeof value === "object" ? JSON.stringify(value) : String(value); }
function readableError(error: unknown) { return error instanceof Error ? error.message : String(error); }
function formatTime(value?: string | null) { if (!value) return "未知时间"; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false }); }
