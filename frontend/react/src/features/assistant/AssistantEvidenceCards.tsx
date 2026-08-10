import { AlertTriangle, ArrowRight, CheckCircle2, CircleHelp, FileText, XCircle } from "lucide-react";
import type { AssistantCitation } from "./types";

type Props = {
  citations: AssistantCitation[];
  onOpenDataset: (id: string) => void;
  onOpenAnalysis: (id: string) => void;
  onOpenReport: (id: string) => void;
};

const reliabilityLabels = {
  verified: "DataMind 审查通过",
  warning: "DataMind 审查警告",
  rejected: "DataMind 审查未通过",
  unverified: "未提供审查状态",
} as const;

const reliabilityIcons = {
  verified: CheckCircle2,
  warning: AlertTriangle,
  rejected: XCircle,
  unverified: CircleHelp,
} as const;

function citationReliability(citation: AssistantCitation) {
  return citation.reliability ?? { status: "unverified" as const, summary: "未提供统计审查状态。" };
}

function ReliabilityBadge({ citation }: { citation: AssistantCitation }) {
  const reliability = citationReliability(citation);
  const Icon = reliabilityIcons[reliability.status];
  return <em className={`assistant-citation-reliability ${reliability.status}`} title={reliability.summary}><Icon size={13} /> {reliabilityLabels[reliability.status]}</em>;
}

export function AssistantEvidenceCards({ citations, onOpenDataset, onOpenAnalysis, onOpenReport }: Props) {
  if (!citations.length) return null;
  const reports = citations.filter((citation) => citation.source_type === "report" && citation.artifact_role === "deliverable").slice(-1);
  const reportIds = new Set(reports.map((citation) => citation.source_id));
  const evidence = citations.filter((citation) => !reportIds.has(citation.source_id));
  return (
    <>
      {reports.length > 0 && (
        <section className="assistant-report-artifacts" aria-label="DataMind 完整报告">
          {reports.map((report) => (
            <button key={report.source_id} type="button" onClick={() => onOpenReport(report.source_id)}>
              <span className="assistant-report-artifact-icon"><FileText size={20} /></span>
              <span className="assistant-report-artifact-copy"><small>DataMind 完整分析报告 · {reliabilityLabels[citationReliability(report).status]}</small><b>{report.label}</b><span>{report.excerpt || "已根据本次对话需求生成，可查看完整图表、结论与验证信息。"}</span></span>
              <span className="assistant-report-artifact-action">查看完整报告 <ArrowRight size={15} /></span>
            </button>
          ))}
        </section>
      )}
      {evidence.length > 0 && (
        <section className="assistant-citations" aria-label="DataMind 证据">
          <strong>支撑证据 · {evidence.length}</strong>
          {evidence.map((citation, index) => (
            <button key={`${citation.source_type}-${citation.source_id}`} type="button" onClick={() => citation.source_type === "analysis_job" ? onOpenAnalysis(citation.source_id) : citation.source_type === "report" ? onOpenReport(citation.source_id) : onOpenDataset(citation.source_id)}>
              <span>{index + 1}</span>
              <div><b>{citation.label}</b><small>{citation.excerpt}</small><ReliabilityBadge citation={citation} /></div>
              <ArrowRight size={14} />
            </button>
          ))}
        </section>
      )}
    </>
  );
}
