import { CircleCheckBig, Network, ShieldCheck, TriangleAlert } from "lucide-react";

export type AnalysisContract = {
  contract_version: string;
  objective: string;
  population: string;
  analysis_type: string;
  metric?: string | null;
  dimensions: string[];
  time_field?: string | null;
  grain: string[];
  method: string;
  assumptions: string[];
  acceptance_criteria: string[];
};

export type StatisticalCheck = {
  code: string;
  status: "passed" | "warning" | "failed" | "not_applicable";
  severity: "info" | "warning" | "error";
  message: string;
};

export type StatisticalVerification = {
  status: "passed" | "warning" | "failed";
  summary: string;
  checks: StatisticalCheck[];
  requires_replan: boolean;
  numeric_evidence_coverage: number;
};

export type AnalysisLineage = {
  nodes: {
    node_id: string;
    node_type: string;
    label: string;
    source_ref?: string;
  }[];
  edges: {
    source_node_id: string;
    target_node_id: string;
    relation: string;
  }[];
  relationship_graph: Record<string, unknown>;
  grain_plan: Record<string, unknown>;
};

export function AnalysisReliabilityPanel({
  contract,
  verification,
  lineage,
}: {
  contract?: AnalysisContract | null;
  verification?: StatisticalVerification | null;
  lineage?: AnalysisLineage | null;
}) {
  if (!contract && !verification && !lineage) return null;
  const status = verification?.status ?? "warning";
  const StatusIcon = status === "passed" ? CircleCheckBig : TriangleAlert;
  const statusClass =
    status === "passed"
      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
      : status === "failed"
        ? "border-rose-200 bg-rose-50 text-rose-800"
        : "border-amber-200 bg-amber-50 text-amber-800";

  return (
    <section className="rounded-lg border border-line bg-white p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <span className="rounded-lg bg-emerald-50 p-2 text-emerald-700">
            <ShieldCheck size={19} />
          </span>
          <div>
            <h4 className="font-black text-slate-950">分析可信度</h4>
            <p className="mt-1 text-sm leading-6 text-slate-600">
              {contract?.method ?? "确定性审查分析证据与统计支持。"}
            </p>
          </div>
        </div>
        {verification && (
          <span className={`inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-black ${statusClass}`}>
            <StatusIcon size={16} />
            {status === "passed" ? "审查通过" : status === "failed" ? "需要重规划" : "存在提示"}
          </span>
        )}
      </div>

      {contract && (
        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <ReliabilityMetric label="分析类型" value={contract.analysis_type} />
          <ReliabilityMetric label="指标" value={contract.metric ?? "计数 / 文本"} />
          <ReliabilityMetric label="分析粒度" value={contract.grain.join(" · ")} />
          <ReliabilityMetric label="总体" value={contract.population} />
        </div>
      )}

      {verification && (
        <div className="mt-5 border-t border-line pt-4">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <b className="text-slate-950">{verification.summary}</b>
            <span className="font-black text-slate-600">
              数值证据覆盖 {Math.round(verification.numeric_evidence_coverage * 100)}%
            </span>
          </div>
          <div className="mt-3 grid gap-2">
            {verification.checks
              .filter((check) => check.status !== "not_applicable")
              .map((check) => (
                <div key={check.code} className="flex items-start gap-2 text-sm leading-6 text-slate-700">
                  <span className={check.status === "failed" ? "text-rose-600" : check.status === "warning" ? "text-amber-600" : "text-emerald-600"}>
                    {check.status === "passed" ? "✓" : "!"}
                  </span>
                  <span>{check.message}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {lineage && (
        <LineageSummary lineage={lineage} />
      )}
    </section>
  );
}

function ReliabilityMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-3">
      <p className="text-xs font-black text-slate-500">{label}</p>
      <p className="mt-1 break-words text-sm font-bold text-slate-950">{value}</p>
    </div>
  );
}

function LineageSummary({ lineage }: { lineage: AnalysisLineage }) {
  const graphEdges = Array.isArray(lineage.relationship_graph.edges)
    ? lineage.relationship_graph.edges
    : [];
  const steps = Array.isArray(lineage.grain_plan.steps)
    ? lineage.grain_plan.steps as Record<string, unknown>[]
    : [];
  const isSafe = lineage.grain_plan.safe !== false;
  const reportCount = lineage.nodes.filter((node) => node.node_type === "report").length;

  return (
    <div className="mt-5 border-t border-line pt-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-2">
          <Network className="mt-0.5 text-emerald-700" size={17} />
          <div>
            <b className="text-sm text-slate-950">字段到报告血缘</b>
            <p className="mt-1 text-sm leading-6 text-slate-600">
              {lineage.nodes.length} 个节点 · {lineage.edges.length} 条依赖 · {graphEdges.length} 条实体关系
            </p>
          </div>
        </div>
        <span className={`rounded-lg border px-2.5 py-1.5 text-xs font-black ${
          isSafe
            ? "border-emerald-200 bg-emerald-50 text-emerald-700"
            : "border-rose-200 bg-rose-50 text-rose-700"
        }`}>
          {isSafe ? "粒度安全" : "已阻断粒度风险"}
        </span>
      </div>
      {!!steps.length && (
        <div className="mt-3 flex flex-wrap gap-2">
          {steps.slice(0, 5).map((step, index) => (
            <span key={String(step.relationship_id ?? index)} className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-bold text-slate-700">
              {String(step.strategy ?? "direct_join")} · {String(step.from_entity_id ?? "")} → {String(step.to_entity_id ?? "")}
            </span>
          ))}
        </div>
      )}
      {reportCount > 0 && (
        <p className="mt-3 text-xs font-bold text-slate-500">
          结论与图表已绑定到本次报告，可追溯字段来源、关系路径和分析粒度。
        </p>
      )}
    </div>
  );
}
