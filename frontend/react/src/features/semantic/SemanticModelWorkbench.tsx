import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Boxes,
  Braces,
  CheckCircle2,
  ChevronRight,
  GitBranch,
  Loader2,
  Plus,
  Save,
  Send,
} from "lucide-react";

import { apiGet, apiPost, apiPut } from "../../api-client";

type SemanticValidation = {
  valid: boolean;
  errors: string[];
  warnings: string[];
  schema_fingerprint: string;
};

type SemanticModel = {
  model_id: string;
  scope_type: "dataset" | "dataset_group";
  scope_id: string;
  name: string;
  version: number;
  revision: number;
  status: "draft" | "published" | "stale" | "archived";
  source: string;
  definition: SemanticDefinition;
  validation?: SemanticValidation | null;
};

type SemanticField = {
  field_id?: string;
  id?: string;
  name?: string;
  source_name?: string;
  type?: string;
  role?: string;
};

type SemanticFieldOption = SemanticField & {
  field_id: string;
  entity_id: string;
  entity_name: string;
};

type SemanticEntity = {
  id: string;
  name: string;
  entity_type?: string;
  grain?: string;
  fields?: SemanticField[];
};

type SemanticDimension = {
  id: string;
  name: string;
  aliases?: string[];
  entity_id?: string;
  field_id?: string;
  type?: string;
  time_grains?: string[];
};

type SemanticMetric = {
  id: string;
  name: string;
  aliases?: string[];
  description?: string;
  unit?: string;
  format?: string;
  formula?: Record<string, unknown>;
};

type SemanticRelationship = {
  id: string;
  left_entity_id?: string;
  right_entity_id?: string;
  join_type?: string;
  cardinality?: string;
  enabled?: boolean;
  risk_note?: string;
};

type SemanticDefinition = Record<string, unknown> & {
  entities?: SemanticEntity[];
  dimensions?: SemanticDimension[];
  metrics?: SemanticMetric[];
  relationships?: SemanticRelationship[];
  unresolved_bindings?: unknown[];
};

type DatasetGroupRef = { group_id: string; name: string };
type EditorView = "visual" | "json";

const AGGREGATIONS = ["sum", "avg", "min", "max", "count", "count_distinct"];

export function SemanticModelWorkbench({ group }: { group: DatasetGroupRef }) {
  const [models, setModels] = useState<SemanticModel[]>([]);
  const [active, setActive] = useState<SemanticModel | null>(null);
  const [definition, setDefinition] = useState<SemanticDefinition>({});
  const [jsonText, setJsonText] = useState("{}");
  const [view, setView] = useState<EditorView>("visual");
  const [validation, setValidation] = useState<SemanticValidation | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const load = async (preferredId?: string) => {
    const payload = await apiGet<{ models: SemanticModel[] }>(
      `/store/semantic-models?scope_type=dataset_group&scope_id=${group.group_id}`,
    );
    setModels(payload.models);
    const selected =
      payload.models.find((item) => item.model_id === preferredId) ??
      payload.models.find((item) => item.status === "draft") ??
      payload.models[0] ??
      null;
    selectModel(selected);
  };

  useEffect(() => {
    void load().catch((error) => setMessage(readableError(error)));
  }, [group.group_id]);

  const selectModel = (model: SemanticModel | null) => {
    setActive(model);
    const next = model?.definition ?? {};
    setDefinition(next);
    setJsonText(JSON.stringify(next, null, 2));
    setValidation(model?.validation ?? null);
  };

  const updateDefinition = (next: SemanticDefinition) => {
    setDefinition(next);
    setJsonText(JSON.stringify(next, null, 2));
    setValidation(null);
    setMessage("");
  };

  const createDraft = async () => {
    await run(async () => {
      const model = await apiPost<SemanticModel>("/store/semantic-models/drafts", {
        scope_type: "dataset_group",
        scope_id: group.group_id,
        name: `${group.name} 语义模型`,
      });
      await load(model.model_id);
      setMessage("已根据当前字段和数据关系生成可视化草稿。");
    });
  };

  const syncJson = () => {
    try {
      const parsed = JSON.parse(jsonText) as SemanticDefinition;
      updateDefinition(parsed);
      setView("visual");
    } catch (error) {
      setMessage(`JSON 无法解析：${readableError(error)}`);
    }
  };

  const save = async () => {
    if (!active || active.status !== "draft") return;
    await run(async () => {
      const model = await apiPut<SemanticModel>(`/store/semantic-models/${active.model_id}`, {
        revision: active.revision,
        name: active.name,
        definition,
      });
      await load(model.model_id);
      setMessage("语义草稿已保存。");
    });
  };

  const validate = async () => {
    if (!active) return;
    await run(async () => {
      if (active.status === "draft") await saveDraftWithoutMessage();
      const result = await apiPost<SemanticValidation>(
        `/store/semantic-models/${active.model_id}/validate`,
        {},
      );
      setValidation(result);
      setMessage(result.valid ? "校验通过，可以发布。" : "校验未通过，请处理下方问题。");
    });
  };

  const publish = async () => {
    if (!active || active.status !== "draft") return;
    await run(async () => {
      await saveDraftWithoutMessage();
      const result = await apiPost<SemanticValidation>(
        `/store/semantic-models/${active.model_id}/validate`,
        {},
      );
      setValidation(result);
      if (!result.valid) throw new Error(result.errors.join("；"));
      const published = await apiPost<SemanticModel>(
        `/store/semantic-models/${active.model_id}/publish`,
        {},
      );
      await load(published.model_id);
      setMessage("语义模型已发布，后续 Planner 将固定引用该版本。");
    });
  };

  const saveDraftWithoutMessage = async () => {
    if (!active || active.status !== "draft") return;
    const model = await apiPut<SemanticModel>(`/store/semantic-models/${active.model_id}`, {
      revision: active.revision,
      name: active.name,
      definition,
    });
    setActive(model);
  };

  const run = async (task: () => Promise<void>) => {
    setBusy(true);
    setMessage("");
    try {
      await task();
    } catch (error) {
      setMessage(readableError(error));
    } finally {
      setBusy(false);
    }
  };

  const editable = active?.status === "draft";
  const unresolvedCount = Array.isArray(definition.unresolved_bindings)
    ? definition.unresolved_bindings.length
    : 0;

  return (
    <section className="mt-5 overflow-hidden rounded-lg border border-indigo-200 bg-white shadow-sm">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-indigo-100 bg-indigo-50/70 px-5 py-4">
        <div className="flex items-start gap-3">
          <span className="rounded-lg bg-indigo-100 p-2 text-indigo-700"><Boxes size={20} /></span>
          <div>
            <h3 className="text-base font-black text-slate-950">语义模型</h3>
            <p className="mt-0.5 text-sm font-semibold text-slate-600">用业务名称管理实体、指标、维度和关系，无需直接编写 SQL。</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="small-button" disabled={busy} onClick={() => void createDraft()}><Plus size={15} />{models.length ? "新建版本" : "自动生成草稿"}</button>
          {editable && <button className="small-button" disabled={busy} onClick={() => void save()}><Save size={15} />保存</button>}
          {active && <button className="small-button bg-slate-700 hover:bg-slate-800" disabled={busy} onClick={() => void validate()}><CheckCircle2 size={15} />校验</button>}
          {editable && <button className="small-button" disabled={busy} onClick={() => void publish()}><Send size={15} />发布</button>}
        </div>
      </header>

      <div className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            {models.map((model) => (
              <button key={model.model_id} type="button" className={`relationship-decision ${active?.model_id === model.model_id ? "is-adopted" : ""}`} onClick={() => selectModel(model)}>
                v{model.version} · {statusLabel(model.status)}
              </button>
            ))}
          </div>
          {active && (
            <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-1">
              <button type="button" className={`rounded-md px-3 py-1.5 text-sm font-black ${view === "visual" ? "bg-white text-emerald-800 shadow-sm" : "text-slate-500"}`} onClick={() => setView("visual")}><Boxes className="mr-1 inline" size={14} />可视化</button>
              <button type="button" className={`rounded-md px-3 py-1.5 text-sm font-black ${view === "json" ? "bg-white text-emerald-800 shadow-sm" : "text-slate-500"}`} onClick={() => setView("json")}><Braces className="mr-1 inline" size={14} />高级 JSON</button>
            </div>
          )}
        </div>

        {!active && <div className="mt-4 rounded-lg border border-dashed border-slate-300 p-8 text-center text-sm font-semibold text-slate-500">生成草稿后，可在这里检查业务实体、指标口径和 Join 风险。</div>}
        {active && view === "visual" && (
          <VisualSemanticEditor definition={definition} editable={editable} onChange={updateDefinition} />
        )}
        {active && view === "json" && (
          <div className="mt-4">
            <textarea className="input min-h-80 font-mono text-sm leading-6" value={jsonText} disabled={!editable} onChange={(event) => setJsonText(event.target.value)} aria-label="语义模型 DSL JSON" />
            {editable && <button type="button" className="small-button mt-3" onClick={syncJson}><ChevronRight size={15} />应用 JSON 并返回可视化</button>}
          </div>
        )}

        {unresolvedCount > 0 && <Notice tone="warning">还有 {unresolvedCount} 个字段绑定未解决，发布前必须处理。</Notice>}
        {validation && <ValidationResult validation={validation} />}
        {message && <Notice tone={/失败|错误|invalid|无法|未通过/i.test(message) ? "error" : "info"}>{busy && <Loader2 className="mr-2 inline animate-spin" size={14} />}{message}</Notice>}
      </div>
    </section>
  );
}

function VisualSemanticEditor({ definition, editable, onChange }: { definition: SemanticDefinition; editable: boolean; onChange: (value: SemanticDefinition) => void }) {
  const entities = definition.entities ?? [];
  const metrics = definition.metrics ?? [];
  const dimensions = definition.dimensions ?? [];
  const relationships = definition.relationships ?? [];
  const entityNames = useMemo(() => new Map(entities.map((item) => [item.id, item.name])), [entities]);
  const fields = useMemo<SemanticFieldOption[]>(
    () => entities.flatMap((entity) =>
      (entity.fields ?? []).flatMap((field) => {
        const fieldId = semanticFieldId(field);
        return fieldId
          ? [{ ...field, field_id: fieldId, entity_id: entity.id, entity_name: entity.name }]
          : [];
      }),
    ),
    [entities],
  );

  const updateCollection = <T,>(key: "entities" | "metrics" | "dimensions" | "relationships", id: string, patch: Partial<T>) => {
    const collection = (definition[key] ?? []) as Array<T & { id: string }>;
    onChange({ ...definition, [key]: collection.map((item) => item.id === id ? { ...item, ...patch } : item) });
  };

  return (
    <div className="mt-5 grid gap-5">
      <section>
        <SectionTitle icon={<Boxes size={17} />} title="业务实体" count={entities.length} hint="事实表承载业务事件，维表提供分类属性。" />
        <div className="grid gap-3 lg:grid-cols-2">
          {entities.map((entity) => (
            <article key={entity.id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-start justify-between gap-3">
                <div><b className="text-sm text-slate-950">{entity.name}</b><small className="mt-1 block text-xs font-semibold text-slate-500">{entity.fields?.length ?? 0} 个字段</small></div>
                <select className="input w-32 py-1.5 text-sm" value={entity.entity_type ?? "unknown"} disabled={!editable} onChange={(event) => updateCollection<SemanticEntity>("entities", entity.id, { entity_type: event.target.value })}>
                  <option value="fact">事实表</option><option value="dimension">维表</option><option value="bridge">桥接表</option><option value="lookup">查找表</option><option value="unknown">待确认</option>
                </select>
              </div>
              <label className="mt-3 block text-xs font-black text-slate-600">数据粒度<input className="input mt-1 py-2 text-sm" value={entity.grain ?? ""} disabled={!editable} onChange={(event) => updateCollection<SemanticEntity>("entities", entity.id, { grain: event.target.value })} /></label>
            </article>
          ))}
        </div>
      </section>

      <section>
        <SectionTitle icon={<GitBranch size={17} />} title="关系与血缘" count={relationships.length} hint="发布前检查基数和重复汇总风险。" />
        {relationships.length ? <div className="grid gap-3">{relationships.map((relationship) => (
          <article key={relationship.id} className="grid items-center gap-3 rounded-lg border border-slate-200 p-4 lg:grid-cols-[1fr_auto_1fr_150px_150px]">
            <EntityBadge name={entityNames.get(relationship.left_entity_id ?? "") ?? "未知实体"} />
            <div className="flex items-center justify-center gap-1 text-slate-400"><span className="h-px w-5 bg-slate-300" /><ChevronRight size={17} /><span className="h-px w-5 bg-slate-300" /></div>
            <EntityBadge name={entityNames.get(relationship.right_entity_id ?? "") ?? "未知实体"} />
            <select className="input py-2 text-sm" value={relationship.cardinality ?? "unknown"} disabled={!editable} onChange={(event) => updateCollection<SemanticRelationship>("relationships", relationship.id, { cardinality: event.target.value })}>
              <option value="one_to_one">1 : 1</option><option value="one_to_many">1 : N</option><option value="many_to_one">N : 1</option><option value="many_to_many">N : N</option><option value="unknown">待确认</option>
            </select>
            <label className="flex items-center gap-2 text-sm font-black text-slate-700"><input type="checkbox" checked={relationship.enabled !== false} disabled={!editable} onChange={(event) => updateCollection<SemanticRelationship>("relationships", relationship.id, { enabled: event.target.checked })} />启用</label>
            {relationship.risk_note && <p className="lg:col-span-5 text-sm font-semibold text-amber-700"><AlertTriangle className="mr-1 inline" size={14} />{relationship.risk_note}</p>}
          </article>
        ))}</div> : <Empty text="当前模型没有跨表关系。" />}
      </section>

      <section>
        <SectionTitle icon={<Boxes size={17} />} title="指标" count={metrics.length} hint="常用聚合可直接编辑，复杂公式仍可在高级 JSON 中维护。" />
        <div className="overflow-auto rounded-lg border border-slate-200">
          <table className="w-full min-w-[900px] border-collapse text-left text-sm">
            <thead className="bg-slate-50 text-xs font-black text-slate-500"><tr><th className="p-3">名称</th><th className="p-3">别名</th><th className="p-3">聚合</th><th className="p-3">来源字段</th><th className="p-3">单位</th><th className="p-3">格式</th></tr></thead>
            <tbody>{metrics.map((metric) => {
              const binding = metricBinding(metric);
              const bindingValue = metricBindingValue(binding.entityId, binding.fieldId);
              const bindingExists = fields.some((field) => metricBindingValue(field.entity_id, field.field_id) === bindingValue);
              return <tr key={metric.id} className="border-t border-slate-200 bg-white">
                <td className="p-3"><input className="input py-2" value={metric.name} disabled={!editable} onChange={(event) => updateCollection<SemanticMetric>("metrics", metric.id, { name: event.target.value })} /></td>
                <td className="p-3"><input className="input py-2" value={(metric.aliases ?? []).join("、")} disabled={!editable} onChange={(event) => updateCollection<SemanticMetric>("metrics", metric.id, { aliases: splitAliases(event.target.value) })} /></td>
                <td className="p-3"><select className="input py-2" value={binding.op} disabled={!editable || !binding.fieldId} onChange={(event) => updateCollection<SemanticMetric>("metrics", metric.id, { formula: replaceMetricBinding(metric.formula, event.target.value, binding.entityId, binding.fieldId) })}>{AGGREGATIONS.map((item) => <option key={item}>{item}</option>)}</select></td>
                <td className="p-3"><select aria-label={`${metric.name} 来源字段`} className="input py-2" value={bindingExists ? bindingValue : ""} disabled={!editable} onChange={(event) => { const [entityId, fieldId] = parseMetricBindingValue(event.target.value); updateCollection<SemanticMetric>("metrics", metric.id, { formula: replaceMetricBinding(metric.formula, binding.op, entityId, fieldId) }); }}><option value="" disabled>未绑定</option>{fields.map((field) => { const optionValue = metricBindingValue(field.entity_id, field.field_id); return <option key={optionValue} value={optionValue}>{field.entity_name} · {field.source_name ?? field.name ?? field.field_id}</option>; })}</select></td>
                <td className="p-3"><input className="input py-2" value={metric.unit ?? ""} disabled={!editable} onChange={(event) => updateCollection<SemanticMetric>("metrics", metric.id, { unit: event.target.value })} /></td>
                <td className="p-3"><select className="input py-2" value={metric.format ?? "number"} disabled={!editable} onChange={(event) => updateCollection<SemanticMetric>("metrics", metric.id, { format: event.target.value })}><option value="number">数值</option><option value="currency">货币</option><option value="percent">百分比</option><option value="integer">整数</option></select></td>
              </tr>;
            })}</tbody>
          </table>
        </div>
      </section>

      <section>
        <SectionTitle icon={<Boxes size={17} />} title="维度" count={dimensions.length} hint="维护用户提问时可识别的业务名称和别名。" />
        <div className="overflow-auto rounded-lg border border-slate-200">
          <table className="w-full min-w-[720px] border-collapse text-left text-sm">
            <thead className="bg-slate-50 text-xs font-black text-slate-500"><tr><th className="p-3">名称</th><th className="p-3">别名</th><th className="p-3">实体</th><th className="p-3">类型</th></tr></thead>
            <tbody>{dimensions.map((dimension) => <tr key={dimension.id} className="border-t border-slate-200 bg-white">
              <td className="p-3"><input className="input py-2" value={dimension.name} disabled={!editable} onChange={(event) => updateCollection<SemanticDimension>("dimensions", dimension.id, { name: event.target.value })} /></td>
              <td className="p-3"><input className="input py-2" value={(dimension.aliases ?? []).join("、")} disabled={!editable} onChange={(event) => updateCollection<SemanticDimension>("dimensions", dimension.id, { aliases: splitAliases(event.target.value) })} /></td>
              <td className="p-3 font-bold text-slate-600">{entityNames.get(dimension.entity_id ?? "") ?? "未知"}</td>
              <td className="p-3"><select className="input py-2" value={dimension.type ?? "categorical"} disabled={!editable} onChange={(event) => updateCollection<SemanticDimension>("dimensions", dimension.id, { type: event.target.value, time_grains: event.target.value === "time" ? ["day", "week", "month", "quarter", "year"] : [] })}><option value="categorical">分类</option><option value="time">时间</option></select></td>
            </tr>)}</tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function metricBinding(metric: SemanticMetric) {
  const formula = metric.formula ?? {};
  const expr = isObject(formula.expr) ? formula.expr : {};
  return { op: typeof formula.op === "string" ? formula.op : "sum", entityId: String(expr.entity_id ?? ""), fieldId: String(expr.field_id ?? "") };
}

function replaceMetricBinding(formula: Record<string, unknown> | undefined, op: string, entityId: string, fieldId: string) {
  if (!fieldId) return formula ?? {};
  return { op, expr: { op: "field", entity_id: entityId, field_id: fieldId } };
}

function semanticFieldId(field: SemanticField) {
  return String(field.field_id ?? field.id ?? "");
}

function metricBindingValue(entityId: string, fieldId: string) {
  return JSON.stringify([entityId, fieldId]);
}

function parseMetricBindingValue(value: string): [string, string] {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (Array.isArray(parsed) && parsed.length === 2) {
      return [String(parsed[0] ?? ""), String(parsed[1] ?? "")];
    }
  } catch {
    // Invalid option values are treated as unbound instead of corrupting the DSL.
  }
  return ["", ""];
}

function isObject(value: unknown): value is Record<string, unknown> { return !!value && typeof value === "object" && !Array.isArray(value); }
function splitAliases(value: string) { return value.split(/[、,，]/).map((item) => item.trim()).filter(Boolean); }
function statusLabel(value: SemanticModel["status"]) { return { draft: "草稿", published: "已发布", stale: "已过期", archived: "已归档" }[value]; }
function readableError(error: unknown) { return error instanceof Error ? error.message : String(error); }

function SectionTitle({ icon, title, count, hint }: { icon: React.ReactNode; title: string; count: number; hint: string }) {
  return <div className="mb-3 flex flex-wrap items-end justify-between gap-2"><div><h4 className="flex items-center gap-2 text-sm font-black text-slate-950">{icon}{title}<span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">{count}</span></h4><p className="mt-1 text-xs font-semibold text-slate-500">{hint}</p></div></div>;
}
function EntityBadge({ name }: { name: string }) { return <div className="rounded-lg border border-indigo-100 bg-indigo-50 px-3 py-2 text-center text-sm font-black text-indigo-950">{name}</div>; }
function Empty({ text }: { text: string }) { return <div className="rounded-lg border border-dashed border-slate-300 p-5 text-center text-sm font-semibold text-slate-500">{text}</div>; }
function Notice({ children, tone }: { children: React.ReactNode; tone: "info" | "warning" | "error" }) { const style = tone === "error" ? "border-red-200 bg-red-50 text-red-800" : tone === "warning" ? "border-amber-200 bg-amber-50 text-amber-800" : "border-sky-200 bg-sky-50 text-sky-800"; return <div className={`mt-4 rounded-lg border px-4 py-3 text-sm font-bold ${style}`}>{children}</div>; }
function ValidationResult({ validation }: { validation: SemanticValidation }) { return <div className={`mt-4 rounded-lg border p-4 ${validation.valid ? "border-emerald-200 bg-emerald-50" : "border-red-200 bg-red-50"}`}><div className="flex items-center gap-2 font-black">{validation.valid ? <CheckCircle2 size={17} className="text-emerald-700" /> : <AlertTriangle size={17} className="text-red-700" />}{validation.valid ? "语义模型校验通过" : `发现 ${validation.errors.length} 个阻塞问题`}</div>{validation.errors.map((item) => <p key={item} className="mt-2 text-sm font-semibold text-red-800">{item}</p>)}{validation.warnings.map((item) => <p key={item} className="mt-2 text-sm font-semibold text-amber-800">{item}</p>)}</div>; }
