import React, { useId, useMemo, useRef } from "react";
import { Download } from "lucide-react";
import { Alert } from "../../components/primitives";
import { AnalysisReliabilityPanel } from "../analysis/AnalysisReliabilityPanel";
import type {
  Chart,
  MultimodalInput,
  Report,
  StructuredReport,
  TextAnalysisResult,
} from "../../domain-types";
import { exportChart } from "./chart-export";

const CHART_SERIES_COLORS = [
  "#5B7DB1",
  "#E07A5F",
  "#2A9D8F",
  "#E9B949",
  "#7A6FA8",
  "#4BA3C7",
  "#D96C88",
  "#78A06A",
  "#D58B4E",
  "#6B8E9E",
] as const;
const CHART_LINE_COLOR = "#4F6FAE";

export function TextAnalysisPanel({ results }: { results: TextAnalysisResult[] }) {
  if (!results.length) return null;
  return (
    <section className="mt-5 rounded-xl border border-emerald-100 bg-emerald-50/40 p-4">
      <h4 className="font-black text-slate-950">文本分析工具箱</h4>
      <div className="mt-3 grid gap-3">
        {results.map((result, index) => {
          const topKeywords = arrayOfRecords(result.summary.top_keywords);
          const groups = arrayOfRecords(result.summary.groups);
          return (
            <div key={`${result.text_column}-${index}`} className="rounded-xl border border-line bg-white p-4 text-sm leading-6">
              <div className="flex flex-wrap items-center gap-2">
                <b>{result.text_column}</b>
                {result.group_column && (
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-black text-slate-600">
                    按 {result.group_column} 分组
                  </span>
                )}
                <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-black text-emerald-700">
                  {result.task}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
                <MetricPill label="可分析文本" value={String(result.summary.non_empty_count ?? 0)} />
                <MetricPill label="空文本" value={String(result.summary.empty_count ?? 0)} />
                <MetricPill label="平均长度" value={String(result.summary.avg_length ?? 0)} />
                <MetricPill label="最长文本" value={String(result.summary.max_length ?? 0)} />
              </div>
              {!!result.insights.length && (
                <ul className="mt-3 list-disc space-y-1 pl-5 text-slate-700">
                  {result.insights.map((insight) => (
                    <li key={insight}>{insight}</li>
                  ))}
                </ul>
              )}
              {!!topKeywords.length && (
                <div className="mt-3">
                  <p className="text-xs font-black uppercase tracking-wide text-slate-500">高频关键词</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {topKeywords.slice(0, 12).map((keyword) => (
                      <span key={String(keyword.keyword)} className="rounded-full bg-slate-100 px-2 py-1 text-xs font-bold text-slate-700">
                        {String(keyword.keyword)} · {String(keyword.count)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {!!groups.length && (
                <details className="mt-3 rounded-lg border border-line bg-slate-50 px-3 py-2">
                  <summary className="cursor-pointer font-black">查看分组文本画像</summary>
                  <DataTable rows={groups} emptyText="暂无分组结果。" />
                </details>
              )}
              <ChartList charts={result.charts} />
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-slate-50 px-3 py-2">
      <p className="text-xs font-bold text-slate-500">{label}</p>
      <p className="text-lg font-black text-slate-950">{value}</p>
    </div>
  );
}


export function MultimodalContextPanel({ inputs, compact = false }: { inputs: MultimodalInput[]; compact?: boolean }) {
  if (!inputs.length) return null;
  return (
    <section className={compact ? "mt-4" : ""}>
      <h3 className={compact ? "text-sm font-black text-slate-700" : "section-title"}>多模态上下文</h3>
      <div className="mt-2 grid gap-2">
        {inputs.map((item, index) => (
          <div key={`${item.title}-${index}`} className="rounded-xl border border-line bg-white px-4 py-3 text-sm leading-6">
            <div className="flex flex-wrap items-center gap-2">
              <b>{item.title}</b>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-black text-slate-600">
                {multimodalKindLabel(item.kind)}
              </span>
              {item.media_type && (
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-black text-slate-600">
                  {item.media_type}
                </span>
              )}
              <span className={`rounded-full px-2 py-0.5 text-xs font-black ${multimodalStatusClass(item.processing_status)}`}>
                {multimodalStatusLabel(item)}
              </span>
            </div>
            <p className="mt-2 text-slate-600">{item.description}</p>
            {item.text_excerpt && (
              <details className="mt-3 rounded-lg border border-line bg-slate-50 px-3 py-2">
                <summary className="cursor-pointer font-black">查看 PDF 抽取文本</summary>
                <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-700">{item.text_excerpt}</pre>
              </details>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

export function ReportVersionCompare({ left, right }: { left: Report; right: Report }) {
  const leftReport = structuredReportFromMetadata(left.metadata);
  const rightReport = structuredReportFromMetadata(right.metadata);
  const rows = [
    {
      项目: "标题",
      对比版本: left.title,
      当前版本: right.title,
      状态: left.title === right.title ? "相同" : "变化",
    },
    {
      项目: "问题",
      对比版本: String(left.metadata?.question ?? ""),
      当前版本: String(right.metadata?.question ?? ""),
      状态: String(left.metadata?.question ?? "") === String(right.metadata?.question ?? "") ? "相同" : "变化",
    },
    {
      项目: "Executive Summary",
      对比版本: leftReport?.executive_summary ?? left.markdown.slice(0, 180),
      当前版本: rightReport?.executive_summary ?? right.markdown.slice(0, 180),
      状态: (leftReport?.executive_summary ?? left.markdown) === (rightReport?.executive_summary ?? right.markdown) ? "相同" : "变化",
    },
    {
      项目: "Key Findings",
      对比版本: String(leftReport?.key_findings?.length ?? 0),
      当前版本: String(rightReport?.key_findings?.length ?? 0),
      状态: (leftReport?.key_findings?.length ?? 0) === (rightReport?.key_findings?.length ?? 0) ? "相同" : "变化",
    },
    {
      项目: "Validation Issues",
      对比版本: String(leftReport?.validation_issues?.length ?? 0),
      当前版本: String(rightReport?.validation_issues?.length ?? 0),
      状态: (leftReport?.validation_issues?.length ?? 0) === (rightReport?.validation_issues?.length ?? 0) ? "相同" : "变化",
    },
    {
      项目: "Recommended Next Steps",
      对比版本: String(leftReport?.recommended_next_steps?.length ?? 0),
      当前版本: String(rightReport?.recommended_next_steps?.length ?? 0),
      状态: (leftReport?.recommended_next_steps?.length ?? 0) === (rightReport?.recommended_next_steps?.length ?? 0) ? "相同" : "变化",
    },
  ];
  return (
    <section className="report-toolbar mb-4 rounded-xl border border-amber-200 bg-amber-50 p-4">
      <h4 className="section-heading">版本对比</h4>
      <p className="mb-3 text-sm font-bold text-slate-600">
        v{left.version ?? 1} 对比 v{right.version ?? 1}
      </p>
      <DataTable rows={rows} emptyText="暂无可对比字段。" />
    </section>
  );
}

export function DataTable({ rows, emptyText }: { rows: Record<string, unknown>[]; emptyText: string }) {
  const columns = useMemo(() => Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 8), [rows]);
  if (!rows.length) return <Alert>{emptyText}</Alert>;
  return (
    <div className="max-h-[360px] overflow-auto rounded-xl border border-line bg-white shadow-[0_10px_26px_rgba(15,23,42,0.05)]">
      <table className="w-full border-collapse text-left text-sm text-slate-800">
        <thead className="sticky top-0 bg-slate-100 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {columns.map((column) => (
              <th key={column} className="border-b border-line px-3 py-3 font-black">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 80).map((row, index) => (
            <tr key={index} className="border-b border-slate-100 odd:bg-white even:bg-slate-50/70">
              {columns.map((column) => (
                <td key={column} className="max-w-[280px] truncate px-3 py-3">
                  {String(row[column] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ChartList({ charts }: { charts: Chart[] }) {
  if (!charts.length) return null;
  return (
    <div className="mt-4 grid min-w-0 gap-4">
      {charts.map((chart, index) => <ChartCard key={`${chart.title}-${index}`} chart={chart} />)}
    </div>
  );
}

function ChartCard({ chart }: { chart: Chart }) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const displayScopeMessage = chartDisplayScopeMessage(chart);
  return (
    <div ref={cardRef} className="chart-card">
      <div className="chart-card-header">
        <div className="min-w-0">
          <p className="chart-card-eyebrow">数据可视化</p>
          <h4 className="chart-card-title">{chart.title}</h4>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2 report-toolbar">
          <span className="chart-type-label">{chartTypeLabel(chart.chart_type)}</span>
          <button type="button" className="small-button h-8 px-2.5" title="导出 SVG" onClick={() => void exportChart(cardRef.current, chart.title, "svg")}><Download size={14} />SVG</button>
          <button type="button" className="small-button h-8 px-2.5" title="导出高清 PNG" onClick={() => void exportChart(cardRef.current, chart.title, "png")}><Download size={14} />PNG</button>
        </div>
      </div>
      <div className="chart-visualization"><ChartVisualization chart={chart} /></div>
      {displayScopeMessage && <p className="chart-explanation">{displayScopeMessage}</p>}
      {chart.explanation && <p className="chart-explanation">{chart.explanation}</p>}
      <details className="chart-data-disclosure"><summary>查看图表数据</summary><DataTable rows={chart.data} emptyText="图表没有数据。" /></details>
    </div>
  );
}

export function StructuredReportPreview({ report }: { report: StructuredReport }) {
  const textAnalysis = textAnalysisFromUnknown(report.python_results?.text_analysis);
  return (
    <div className="mt-5 space-y-5">
      <section className="rounded-xl border border-sky-200 bg-sky-50 p-5 text-sm leading-7 text-slate-950">{report.executive_summary}</section>
      <AnalysisReliabilityPanel
        contract={report.analysis_contract}
        verification={report.statistical_verification}
        lineage={report.analysis_lineage}
      />
      {report.analysis_context && (
        <section className="rounded-xl border border-line bg-white p-4 text-sm leading-6 text-slate-700">
          <p className="font-black text-slate-950">分析上下文</p>
          <p className="mt-2">{report.analysis_context}</p>
        </section>
      )}
      <TextAnalysisPanel results={textAnalysis} />
      {!!report.key_findings?.length && (
        <section>
          <h4 className="section-heading">核心发现</h4>
          <div className="grid gap-3">
            {report.key_findings.map((finding) => (
              <div key={finding.title} className="rounded-xl border border-line bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)]">
                <p className="font-black">{finding.title}</p>
                <p className="mt-2 text-sm leading-6">{finding.content}</p>
                {(finding.evidence || finding.business_impact || finding.recommended_action) && (
                  <div className="mt-3 grid gap-2 text-xs leading-5 text-slate-600">
                    {finding.evidence && <span>证据：{finding.evidence}</span>}
                    {finding.business_impact && <span>影响：{finding.business_impact}</span>}
                    {finding.recommended_action && <span>建议：{finding.recommended_action}</span>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
      <ChartList charts={report.charts ?? []} />
      {!!report.recommended_next_steps?.length && (
        <section>
          <h4 className="section-heading">建议下一步</h4>
          <div className="grid gap-2">
            {report.recommended_next_steps.map((step, index) => (
              <div key={`${step}-${index}`} className="rounded-xl border border-line bg-white px-4 py-3 text-sm">
                {step}
              </div>
            ))}
          </div>
        </section>
      )}
      {!!report.sql_results?.length && (
        <section>
          <h4 className="section-heading">SQL 结果</h4>
          <DataTable rows={report.sql_results} emptyText="没有 SQL 结果。" />
        </section>
      )}
      {!!report.data_gaps?.length && (
        <section>
          <h4 className="section-heading">数据缺口</h4>
          <div className="grid gap-2">
            {report.data_gaps.map((gap) => (
              <Alert key={gap}>{gap}</Alert>
            ))}
          </div>
        </section>
      )}
      {!!report.validation_issues?.length && (
        <section>
          <h4 className="section-heading">校验问题</h4>
          <div className="grid gap-2">
            {report.validation_issues.map((issue, index) => (
              <div key={`${issue.finding_ref}-${index}`} className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6">
                <b>{issue.severity}</b> · {issue.finding_ref}: {issue.issue}
                {issue.suggestion && <p className="text-slate-600">{issue.suggestion}</p>}
              </div>
            ))}
          </div>
        </section>
      )}
      {!!report.analysis_trace?.length && (
        <section>
          <h4 className="section-heading">分析轨迹</h4>
          <div className="grid gap-2">
            {report.analysis_trace.map((round) => (
              <TraceRoundCard key={round.round_number} round={round} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function TraceRoundCard({ round }: { round: NonNullable<StructuredReport["analysis_trace"]>[number] }) {
  const execution = round.execution_result ?? {};
  const fanoutMode = String(execution.fanout_mode ?? "serial");
  const isFanout = fanoutMode.includes("fanout");
  return (
    <div className="rounded-xl border border-line bg-white px-4 py-3 text-sm leading-6">
      <div className="flex flex-wrap items-center gap-2">
        <p className="font-black">Round {round.round_number}</p>
        <span className={`rounded-full px-2 py-0.5 text-xs font-black ${round.validation_status === "warning" ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-700"}`}>
          {round.validation_status ?? "passed"}
        </span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-black ${isFanout ? "bg-violet-50 text-violet-700" : "bg-slate-100 text-slate-600"}`}>
          {isFanout ? "并行探索" : "串行基础"}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-5">
        <MetricPill label="路线" value={round.plan?.route ?? "analysis"} />
        <MetricPill label="SQL 行数" value={metricValue(execution.sql_row_count)} />
        <MetricPill label="图表" value={metricValue(execution.chart_count)} />
        <MetricPill label="文本分析" value={metricValue(execution.text_analysis_count)} />
        <MetricPill label="Python" value={String(execution.python_source ?? "-")} />
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-xs font-bold text-slate-500">
        {round.plan?.metric_column && <span>指标：{round.plan.metric_column}</span>}
        {round.plan?.category_column && <span>维度：{round.plan.category_column}</span>}
        {round.plan?.time_column && <span>时间：{round.plan.time_column}</span>}
        {Boolean(execution.fanout_group) && <span>分支组：{String(execution.fanout_group)}</span>}
      </div>
      <p className="mt-3 font-bold text-slate-950">{round.hypothesis?.statement}</p>
      <p className="text-slate-600">{round.reflection?.insight_text}</p>
      {Boolean(execution.python_execution_error) && (
        <p className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700">
          Python 回退：{String(execution.python_execution_error)}
        </p>
      )}
      {executionIssues(execution, "plan_validation_issues").map((issue, index) => (
        <p key={`${round.round_number}-plan-issue-${index}`} className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
          Plan Harness：{issue.finding_ref ? `${issue.finding_ref} · ` : ""}
          {issue.issue}
        </p>
      ))}
    </div>
  );
}

function ChartVisualization({ chart }: { chart: Chart }) {
  if (!chart.data.length) return <Alert>图表没有数据。</Alert>;
  if (chart.chart_type === "pie") return <PieChartSvg chart={chart} />;
  if (chart.chart_type === "line") return <CartesianChartSvg chart={chart} mode="line" />;
  if (chart.chart_type === "histogram") return <HistogramChartSvg chart={chart} />;
  if (chart.chart_type === "box_plot") return <BoxPlotSvg chart={chart} />;
  if (chart.chart_type === "correlation_heatmap") return <HeatmapSvg chart={chart} />;
  return <CartesianChartSvg chart={chart} mode="bar" />;
}

function CartesianChartSvg({ chart, mode }: { chart: Chart; mode: "bar" | "line" }) {
  const chartId = useId().replace(/:/g, "");
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "x");
  const cartesianKeys = Object.keys(chart.data[0] ?? {});
  const yKey = String(chart.spec.y ?? cartesianKeys[cartesianKeys.length - 1] ?? "y");
  const visibleRows = mode === "bar" && chart.data.length > 24
    ? [...chart.data].sort((leftRow, rightRow) => numberValue(rightRow[yKey]) - numberValue(leftRow[yKey]))
    : chart.data;
  const points = visibleRows
    .slice(0, 24)
    .map((row) => ({ label: String(row[xKey] ?? ""), value: numberValue(row[yKey]) }))
    .filter((point) => Number.isFinite(point.value));
  if (!points.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  const width = 760;
  const height = 300;
  const left = 64;
  const right = 28;
  const top = 42;
  const bottom = 58;
  const max = Math.max(...points.map((point) => point.value), 1);
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const step = innerWidth / Math.max(points.length, 1);
  const ticks = axisTicks(max);
  const coordinates = points.map((point, index) => ({
    ...point,
    x: left + step * index + step / 2,
    y: height - bottom - (point.value / max) * innerHeight,
  }));
  const linePath = coordinates.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const areaPath = `${linePath} L ${coordinates[coordinates.length - 1].x} ${height - bottom} L ${coordinates[0].x} ${height - bottom} Z`;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={chart.title}>
      <defs>
        <linearGradient id={`${chartId}-area`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={CHART_LINE_COLOR} stopOpacity="0.24" />
          <stop offset="100%" stopColor={CHART_LINE_COLOR} stopOpacity="0" />
        </linearGradient>
        <filter id={`${chartId}-shadow`} x="-20%" y="-20%" width="140%" height="160%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" floodColor="#334155" floodOpacity="0.14" />
        </filter>
      </defs>
      <text x={left} y={22} fontSize="11" fontWeight="700" fill="#64748b">{yKey}</text>
      {ticks.map((tick) => {
        const y = height - bottom - (tick / max) * innerHeight;
        return (
          <g key={`tick-${tick}`}>
            <line x1={left} y1={y} x2={width - right} y2={y} stroke="#dbe4ee" strokeDasharray="4 6" />
            <text x={left - 12} y={y + 4} textAnchor="end" fontSize="10" fontWeight="600" fill="#64748b">
              {formatAxisValue(tick)}
            </text>
          </g>
        );
      })}
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} stroke="#bac8d8" />
      {mode === "bar"
        ? coordinates.map((point, index) => {
            const barWidth = Math.min(Math.max(step * 0.58, 8), 46);
            const barHeight = (point.value / max) * innerHeight;
            const x = point.x - barWidth / 2;
            const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
            return (
              <rect key={`${point.label}-${index}`} x={x} y={point.y} width={barWidth} height={barHeight} rx={7} fill={color} fillOpacity={0.94} filter={`url(#${chartId}-shadow)`}>
                <title>{`${point.label}: ${formatAxisValue(point.value)}`}</title>
              </rect>
            );
          })
        : (
            <>
              <path d={areaPath} fill={`url(#${chartId}-area)`} />
              <path d={linePath} fill="none" stroke={CHART_LINE_COLOR} strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round" />
              {coordinates.map((point, index) => (
                <circle key={`${point.label}-${index}`} cx={point.x} cy={point.y} r={4.5} fill="white" stroke={CHART_LINE_COLOR} strokeWidth={3}>
                  <title>{`${point.label}: ${formatAxisValue(point.value)}`}</title>
                </circle>
              ))}
            </>
          )}
      {coordinates.map((point, index) => (
        <text key={`${point.label}-label-${index}`} x={point.x} y={height - 22} textAnchor="middle" fontSize="10" fontWeight="600" fill="#526174">
          {shortLabel(point.label)}
        </text>
      ))}
    </svg>
  );
}

function PieChartSvg({ chart }: { chart: Chart }) {
  const nameKey = String(chart.spec.names ?? Object.keys(chart.data[0] ?? {})[0] ?? "name");
  const pieKeys = Object.keys(chart.data[0] ?? {});
  const valueKey = String(chart.spec.values ?? pieKeys[pieKeys.length - 1] ?? "value");
  const slices = chart.data
    .slice(0, 8)
    .map((row) => ({ label: String(row[nameKey] ?? ""), value: Math.max(numberValue(row[valueKey]), 0) }))
    .filter((slice) => slice.value > 0);
  const total = slices.reduce((sum, slice) => sum + slice.value, 0);
  if (!total) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  let current = 0;
  return (
    <div className="chart-pie-layout">
      <svg viewBox="0 0 220 220" className="chart-pie-svg" role="img" aria-label={chart.title}>
        {slices.map((slice, index) => {
          const start = current;
          const end = current + (slice.value / total) * Math.PI * 2;
          current = end;
          return (
            <path key={slice.label} d={arcPath(110, 110, 82, start, end)} fill={CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]} stroke="white" strokeWidth="3" strokeLinejoin="round">
              <title>{`${slice.label}: ${formatAxisValue(slice.value)} (${((slice.value / total) * 100).toFixed(1)}%)`}</title>
            </path>
          );
        })}
        <circle cx="110" cy="110" r="48" fill="white" />
        <text x="110" y="104" textAnchor="middle" fontSize="11" fontWeight="700" fill="#64748b">合计</text>
        <text x="110" y="124" textAnchor="middle" fontSize="16" fontWeight="800" fill="#0f172a">{formatAxisValue(total)}</text>
      </svg>
      <div className="chart-legend">
        {slices.map((slice, index) => (
          <div key={slice.label} className="chart-legend-row">
            <span className="flex min-w-0 items-center gap-2.5">
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length] }} />
              <span className="truncate font-bold text-slate-700">{slice.label}</span>
            </span>
            <span className="flex shrink-0 items-baseline gap-2">
              <b className="text-slate-950">{formatAxisValue(slice.value)}</b>
              <span className="text-xs font-bold text-slate-500">{((slice.value / total) * 100).toFixed(1)}%</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HistogramChartSvg({ chart }: { chart: Chart }) {
  if (chart.spec.y) return <CartesianChartSvg chart={chart} mode="bar" />;
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "value");
  const values = chart.data.map((row) => numberValue(row[xKey])).filter(Number.isFinite);
  if (!values.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bucketCount = 10;
  const buckets = Array.from({ length: bucketCount }, (_, index) => ({ label: `${index + 1}`, value: 0 }));
  values.forEach((value) => {
    const ratio = max === min ? 0 : (value - min) / (max - min);
    buckets[Math.min(bucketCount - 1, Math.floor(ratio * bucketCount))].value += 1;
  });
  return <CartesianChartSvg chart={{ ...chart, chart_type: "bar", spec: { x: "label", y: "value" }, data: buckets }} mode="bar" />;
}

function BoxPlotSvg({ chart }: { chart: Chart }) {
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "category");
  const boxKeys = Object.keys(chart.data[0] ?? {});
  const yKey = String(chart.spec.y ?? boxKeys[boxKeys.length - 1] ?? "value");
  const groups = new Map<string, number[]>();
  chart.data.forEach((row) => {
    const label = String(row[xKey] ?? "未分组");
    const value = numberValue(row[yKey]);
    if (Number.isFinite(value)) groups.set(label, [...(groups.get(label) ?? []), value]);
  });
  const summaries = Array.from(groups.entries()).slice(0, 8).map(([label, values]) => ({ label, ...quartiles(values) }));
  if (!summaries.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数值。" />;
  const width = 760;
  const height = 300;
  const left = 64;
  const right = 28;
  const top = 42;
  const bottom = 58;
  const max = Math.max(...summaries.map((item) => item.max), 1);
  const scaleY = (value: number) => height - bottom - (value / max) * (height - top - bottom);
  const step = (width - left - right) / summaries.length;
  const ticks = axisTicks(max);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={chart.title}>
      <text x={left} y={22} fontSize="11" fontWeight="700" fill="#64748b">{yKey}</text>
      {ticks.map((tick) => {
        const y = scaleY(tick);
        return (
          <g key={`box-tick-${tick}`}>
            <line x1={left} y1={y} x2={width - right} y2={y} stroke="#dbe4ee" strokeDasharray="4 6" />
            <text x={left - 12} y={y + 4} textAnchor="end" fontSize="10" fontWeight="600" fill="#64748b">
              {formatAxisValue(tick)}
            </text>
          </g>
        );
      })}
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} stroke="#bac8d8" />
      {summaries.map((item, index) => {
        const x = left + step * index + step / 2;
        const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
        return (
          <g key={item.label}>
            <line x1={x} x2={x} y1={scaleY(item.min)} y2={scaleY(item.max)} stroke={color} strokeWidth={2.5} />
            <line x1={x - 12} x2={x + 12} y1={scaleY(item.min)} y2={scaleY(item.min)} stroke={color} strokeWidth={2.5} />
            <line x1={x - 12} x2={x + 12} y1={scaleY(item.max)} y2={scaleY(item.max)} stroke={color} strokeWidth={2.5} />
            <rect x={x - 22} y={scaleY(item.q3)} width={44} height={Math.max(scaleY(item.q1) - scaleY(item.q3), 3)} rx={6} fill={color} fillOpacity={0.28} stroke={color} strokeWidth={1.8}>
              <title>{`${item.label} · 最小 ${formatAxisValue(item.min)} · 中位数 ${formatAxisValue(item.median)} · 最大 ${formatAxisValue(item.max)}`}</title>
            </rect>
            <line x1={x - 22} x2={x + 22} y1={scaleY(item.median)} y2={scaleY(item.median)} stroke="#0f172a" strokeWidth={3} />
            <text x={x} y={height - 22} textAnchor="middle" fontSize="10" fontWeight="600" fill="#526174">{shortLabel(item.label)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function HeatmapSvg({ chart }: { chart: Chart }) {
  const labels = Array.from(new Set(chart.data.flatMap((row) => [String(row.source ?? ""), String(row.target ?? "")]))).filter(Boolean).slice(0, 10);
  if (!labels.length) return <DataTable rows={chart.data} emptyText="图表没有可绘制的数据。" />;
  const cell = 44;
  const pad = 90;
  const size = pad + labels.length * cell + 16;
  const valueFor = (source: string, target: string) => numberValue(chart.data.find((row) => row.source === source && row.target === target)?.value);
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="chart-svg chart-heatmap-svg" role="img" aria-label={chart.title}>
      {labels.map((label, index) => (
        <React.Fragment key={label}>
          <text x={pad + index * cell + cell / 2} y={pad - 12} textAnchor="middle" fontSize="10" fontWeight="600" fill="#526174">{shortLabel(label)}</text>
          <text x={pad - 10} y={pad + index * cell + cell / 2 + 4} textAnchor="end" fontSize="10" fontWeight="600" fill="#526174">{shortLabel(label)}</text>
        </React.Fragment>
      ))}
      {labels.flatMap((source, yIndex) =>
        labels.map((target, xIndex) => {
          const value = valueFor(source, target);
          const intensity = Math.min(Math.abs(value), 1);
          return (
            <rect
              key={`${source}-${target}`}
              x={pad + xIndex * cell}
              y={pad + yIndex * cell}
              width={cell - 4}
              height={cell - 4}
              rx={6}
              fill={value >= 0 ? `rgba(13, 148, 136, ${0.1 + intensity * 0.82})` : `rgba(225, 29, 72, ${0.1 + intensity * 0.82})`}
              stroke="rgba(255,255,255,0.9)"
            >
              <title>{`${source} × ${target}: ${formatAxisValue(value)}`}</title>
            </rect>
          );
        }),
      )}
    </svg>
  );
}

export function structuredReportFromMetadata(metadata: Record<string, unknown>): StructuredReport | null {
  return structuredReportFromUnknown(metadata.structured_report);
}

export function structuredReportFromUnknown(value: unknown): StructuredReport | null {
  if (!value || typeof value !== "object") return null;
  return value as StructuredReport;
}

function executionIssues(
  executionResult: Record<string, unknown> | undefined,
  key: string,
): { severity?: string; finding_ref?: string; issue: string; suggestion?: string }[] {
  const value = executionResult?.[key];
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      severity: typeof item.severity === "string" ? item.severity : undefined,
      finding_ref: typeof item.finding_ref === "string" ? item.finding_ref : undefined,
      issue: typeof item.issue === "string" ? item.issue : "",
      suggestion: typeof item.suggestion === "string" ? item.suggestion : undefined,
    }))
    .filter((item) => item.issue);
}

export function arrayOfRecords(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object");
}

function metricValue(value: unknown): string {
  return typeof value === "number" ? String(value) : "-";
}

function axisTicks(max: number) {
  const upper = Math.max(max, 1);
  return [0, 0.25, 0.5, 0.75, 1].map((ratio) => upper * ratio);
}

function formatAxisValue(value: number) {
  if (Math.abs(value) >= 1000) return value.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(value < 10 ? 1 : 0);
}

function chartTypeLabel(chartType: Chart["chart_type"]) {
  const labels: Record<string, string> = {
    bar: "柱状图",
    line: "趋势图",
    pie: "占比图",
    histogram: "分布图",
    box_plot: "箱线图",
    correlation_heatmap: "相关性热力图",
  };
  return labels[chartType] ?? chartType;
}

function textAnalysisFromUnknown(value: unknown): TextAnalysisResult[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      task: typeof item.task === "string" ? item.task : "text_analysis",
      text_column: typeof item.text_column === "string" ? item.text_column : "text",
      group_column: typeof item.group_column === "string" ? item.group_column : null,
      summary: Boolean(item.summary) && typeof item.summary === "object" ? item.summary as Record<string, unknown> : {},
      insights: Array.isArray(item.insights) ? item.insights.map(String) : [],
      charts: Array.isArray(item.charts) ? item.charts as Chart[] : [],
    }));
}

export function multimodalInputsFromMetadata(metadata: Record<string, unknown>): MultimodalInput[] {
  const value = metadata.multimodal_inputs;
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      kind: isMultimodalKind(item.kind) ? item.kind : "note",
      title: typeof item.title === "string" ? item.title : "未命名上下文",
      description: typeof item.description === "string" ? item.description : "",
      source_ref: typeof item.source_ref === "string" ? item.source_ref : null,
      media_type: typeof item.media_type === "string" ? item.media_type : null,
      data_url: typeof item.data_url === "string" ? item.data_url : null,
      processing_status: typeof item.processing_status === "string" ? item.processing_status : null,
      text_excerpt: typeof item.text_excerpt === "string" ? item.text_excerpt : null,
    }))
    .filter((item) => item.description);
}

function isMultimodalKind(value: unknown): value is MultimodalInput["kind"] {
  return value === "image" || value === "chart" || value === "pdf_page" || value === "screenshot" || value === "note";
}

export function htmlReportForDownload(title: string, report: StructuredReport) {
  const findings = (report.key_findings ?? [])
    .map((finding) => `<section class="card"><h2>${escapeHtml(finding.title)}</h2><p>${escapeHtml(finding.content)}</p>${finding.evidence ? `<p class="muted">证据：${escapeHtml(finding.evidence)}</p>` : ""}${finding.recommended_action ? `<p class="muted">建议：${escapeHtml(finding.recommended_action)}</p>` : ""}</section>`)
    .join("");
  const charts = (report.charts ?? [])
    .map((chart) => {
      const displayScopeMessage = chartDisplayScopeMessage(chart);
      return `<section class="card"><h2>${escapeHtml(chart.title)}</h2><p class="muted">${escapeHtml(chart.chart_type)}</p>${chartSvgMarkup(chart)}${displayScopeMessage ? `<p>${escapeHtml(displayScopeMessage)}</p>` : ""}${chart.explanation ? `<p>${escapeHtml(chart.explanation)}</p>` : ""}</section>`;
    })
    .join("");
  const sql = rowsTableMarkup(report.sql_results ?? []);
  const gaps = (report.data_gaps ?? []).map((gap) => `<li>${escapeHtml(gap)}</li>`).join("");
  const issues = (report.validation_issues ?? []).map((issue) => `<li><b>${escapeHtml(issue.severity)}</b> · ${escapeHtml(issue.finding_ref)}: ${escapeHtml(issue.issue)}</li>`).join("");
  const trace = (report.analysis_trace ?? []).map((round) => {
    const summary = [
      round.plan?.route ? `route: ${round.plan.route}` : "",
      round.plan?.metric_column ? `metric: ${round.plan.metric_column}` : "",
      round.plan?.category_column ? `dimension: ${round.plan.category_column}` : "",
      typeof round.execution_result?.sql_row_count === "number" ? `sql rows: ${round.execution_result.sql_row_count}` : "",
      typeof round.execution_result?.chart_count === "number" ? `charts: ${round.execution_result.chart_count}` : "",
      round.execution_result?.python_source ? `python: ${String(round.execution_result.python_source)}` : "",
      round.execution_result?.fanout_mode ? `mode: ${String(round.execution_result.fanout_mode)}` : "",
    ].filter(Boolean).join(" · ");
    return `<section class="card"><h2>Round ${round.round_number}</h2>${summary ? `<p class="muted">${escapeHtml(summary)}</p>` : ""}<p>${escapeHtml(round.hypothesis?.statement ?? "")}</p><p class="muted">${escapeHtml(round.reflection?.insight_text ?? "")}</p></section>`;
  }).join("");
  const nextSteps = (report.recommended_next_steps ?? []).map((step) => `<li>${escapeHtml(step)}</li>`).join("");
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <style>
    body{margin:0;background:#f3f6fb;color:#111827;font-family:Arial,"Microsoft YaHei",sans-serif}
    main{max-width:1080px;margin:0 auto;padding:40px 28px 64px}
    h1{font-size:34px;margin:0 0 20px}.card{background:#fff;border:1px solid #dbe5f0;border-radius:8px;padding:20px;margin:14px 0}
    .summary{padding:24px 28px;background:#cfe6fb;border-radius:8px;font-size:17px;line-height:1.75}
    .muted{color:#64748b}.chart-grid{display:grid;grid-template-columns:280px 1fr;gap:20px;align-items:center}
    .legend{display:grid;gap:8px;font-size:14px}table{width:100%;border-collapse:collapse;background:white}
    th,td{padding:10px 12px;border-bottom:1px solid #e5edf5;text-align:left;font-size:14px}th{background:#111827;color:white}
  </style>
</head>
<body>
  <main>
    <h1>${escapeHtml(title)}</h1>
    <section class="summary">${escapeHtml(report.executive_summary)}</section>
    ${report.analysis_context ? `<h2>Analysis Context</h2><section class="card">${escapeHtml(report.analysis_context)}</section>` : ""}
    <h2>Key Findings</h2>${findings || "<p class='muted'>暂无核心发现。</p>"}
    <h2>Visualizations</h2>${charts || "<p class='muted'>暂无图表。</p>"}
    <h2>SQL Results</h2>${sql || "<p class='muted'>暂无 SQL 结果。</p>"}
    <h2>Data Gaps</h2>${gaps ? `<ul>${gaps}</ul>` : "<p class='muted'>暂无数据缺口。</p>"}
    <h2>Validation Issues</h2>${issues ? `<ul>${issues}</ul>` : "<p class='muted'>暂无校验问题。</p>"}
    <h2>Recommended Next Steps</h2>${nextSteps ? `<ul>${nextSteps}</ul>` : "<p class='muted'>暂无建议。</p>"}
    <h2>Analysis Trace</h2>${trace || "<p class='muted'>暂无分析轨迹。</p>"}
  </main>
</body>
</html>`;
}

function chartSvgMarkup(chart: Chart) {
  if (!chart.data.length) return "<p class='muted'>暂无图表数据。</p>";
  if (chart.chart_type === "pie") return pieSvgMarkup(chart);
  if (chart.chart_type === "line") return cartesianSvgMarkup(chart, "line");
  if (chart.chart_type === "histogram") return histogramSvgMarkup(chart);
  if (chart.chart_type === "box_plot") return boxSvgMarkup(chart);
  if (chart.chart_type === "correlation_heatmap") return heatmapSvgMarkup(chart);
  return cartesianSvgMarkup(chart, "bar");
}

function cartesianSvgMarkup(chart: Chart, mode: "bar" | "line") {
  const keys = Object.keys(chart.data[0] ?? {});
  const xKey = String(chart.spec.x ?? keys[0] ?? "x");
  const yKey = String(chart.spec.y ?? keys[keys.length - 1] ?? "y");
  const visibleRows = mode === "bar" && chart.data.length > 24
    ? [...chart.data].sort((leftRow, rightRow) => numberValue(rightRow[yKey]) - numberValue(leftRow[yKey]))
    : chart.data;
  const points = visibleRows
    .slice(0, 24)
    .map((row) => ({ label: String(row[xKey] ?? ""), value: numberValue(row[yKey]) }))
    .filter((point) => Number.isFinite(point.value));
  if (!points.length) return rowsTableMarkup(chart.data);
  const width = 720;
  const height = 280;
  const pad = 58;
  const max = Math.max(...points.map((point) => point.value), 1);
  const innerHeight = height - pad * 2;
  const step = (width - pad * 2) / Math.max(points.length, 1);
  let body = axisTicks(max)
    .map((tick) => {
      const y = height - pad - (tick / max) * innerHeight;
      return `<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="#e2e8f0"/><text x="${pad - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${escapeHtml(formatAxisValue(tick))}</text>`;
    })
    .join("");
  body += `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#cbd5e1"/>`;
  if (mode === "line") {
    const polyline = points
      .map((point, index) => `${pad + step * index + step / 2},${height - pad - (point.value / max) * innerHeight}`)
      .join(" ");
    body += `<polyline points="${polyline}" fill="none" stroke="${CHART_LINE_COLOR}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>`;
  } else {
    body += points
      .map((point, index) => {
        const barWidth = Math.max(step * 0.58, 6);
        const barHeight = (point.value / max) * innerHeight;
        const x = pad + step * index + (step - barWidth) / 2;
        const y = height - pad - barHeight;
        const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
        return `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="5" fill="${color}" fill-opacity="0.94"/>`;
      })
      .join("");
  }
  body += points
    .map((point, index) => `<text x="${pad + step * index + step / 2}" y="${height - 14}" text-anchor="middle" font-size="11" fill="#475569">${escapeHtml(shortLabel(point.label))}</text>`)
    .join("");
  return svgMarkup(width, height, body);
}

function chartDisplayScopeMessage(chart: Chart) {
  if (chart.chart_type !== "bar" || chart.data.length <= 24) return "";
  const keys = Object.keys(chart.data[0] ?? {});
  const yKey = String(chart.spec.y ?? keys[keys.length - 1] ?? "指标");
  return `图中按 ${yKey} 从高到低展示前 24 / ${chart.data.length} 个类别；完整结果见“查看图表数据”。`;
}

function pieSvgMarkup(chart: Chart) {
  const keys = Object.keys(chart.data[0] ?? {});
  const nameKey = String(chart.spec.names ?? keys[0] ?? "name");
  const valueKey = String(chart.spec.values ?? keys[keys.length - 1] ?? "value");
  const slices = chart.data
    .slice(0, 8)
    .map((row) => ({ label: String(row[nameKey] ?? ""), value: Math.max(numberValue(row[valueKey]), 0) }))
    .filter((slice) => slice.value > 0);
  const total = slices.reduce((sum, slice) => sum + slice.value, 0);
  if (!total) return rowsTableMarkup(chart.data);
  let current = 0;
  const paths = slices
    .map((slice, index) => {
      const start = current;
      const end = current + (slice.value / total) * Math.PI * 2;
      current = end;
      return `<path d="${arcPath(110, 110, 82, start, end)}" fill="${CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]}" stroke="white" stroke-width="3"/>`;
    })
    .join("");
  const legend = slices
    .map((slice, index) => `<div><span style="display:inline-block;width:11px;height:11px;background:${CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]};margin-right:8px"></span>${escapeHtml(slice.label)} <b>${((slice.value / total) * 100).toFixed(1)}%</b></div>`)
    .join("");
  return `<div class="chart-grid">${svgMarkup(220, 220, `${paths}<circle cx="110" cy="110" r="44" fill="white"/>`)}<div class="legend">${legend}</div></div>`;
}

function histogramSvgMarkup(chart: Chart) {
  if (chart.spec.y) return cartesianSvgMarkup(chart, "bar");
  const xKey = String(chart.spec.x ?? Object.keys(chart.data[0] ?? {})[0] ?? "value");
  const values = chart.data.map((row) => numberValue(row[xKey])).filter(Number.isFinite);
  if (!values.length) return rowsTableMarkup(chart.data);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const buckets = Array.from({ length: 10 }, (_, index) => ({ label: String(index + 1), value: 0 }));
  values.forEach((value) => {
    const ratio = max === min ? 0 : (value - min) / (max - min);
    buckets[Math.min(9, Math.floor(ratio * 10))].value += 1;
  });
  return cartesianSvgMarkup({ ...chart, spec: { x: "label", y: "value" }, data: buckets }, "bar");
}

function boxSvgMarkup(chart: Chart) {
  const keys = Object.keys(chart.data[0] ?? {});
  const xKey = String(chart.spec.x ?? keys[0] ?? "category");
  const yKey = String(chart.spec.y ?? keys[keys.length - 1] ?? "value");
  const groups = new Map<string, number[]>();
  chart.data.forEach((row) => {
    const value = numberValue(row[yKey]);
    if (Number.isFinite(value)) groups.set(String(row[xKey] ?? "未分组"), [...(groups.get(String(row[xKey] ?? "未分组")) ?? []), value]);
  });
  const summaries = Array.from(groups.entries()).slice(0, 8).map(([label, values]) => ({ label, ...quartiles(values) }));
  if (!summaries.length) return rowsTableMarkup(chart.data);
  const width = 720;
  const height = 280;
  const pad = 58;
  const max = Math.max(...summaries.map((item) => item.max), 1);
  const step = (width - pad * 2) / summaries.length;
  const sy = (value: number) => height - pad - (value / max) * (height - pad * 2);
  let body = axisTicks(max)
    .map((tick) => {
      const y = sy(tick);
      return `<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="#e2e8f0"/><text x="${pad - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b">${escapeHtml(formatAxisValue(tick))}</text>`;
    })
    .join("");
  body += `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#cbd5e1"/>`;
  body += summaries
    .map((item, index) => {
      const x = pad + step * index + step / 2;
      const color = CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length];
      return `<line x1="${x}" x2="${x}" y1="${sy(item.min)}" y2="${sy(item.max)}" stroke="${color}" stroke-width="3"/><rect x="${x - 20}" y="${sy(item.q3)}" width="40" height="${Math.max(sy(item.q1) - sy(item.q3), 2)}" fill="${color}" fill-opacity="0.28" stroke="${color}"/><line x1="${x - 24}" x2="${x + 24}" y1="${sy(item.median)}" y2="${sy(item.median)}" stroke="#111827" stroke-width="3"/><text x="${x}" y="${height - 14}" text-anchor="middle" font-size="11" fill="#475569">${escapeHtml(shortLabel(item.label))}</text>`;
    })
    .join("");
  return svgMarkup(width, height, body);
}

function heatmapSvgMarkup(chart: Chart) {
  const labels = Array.from(new Set(chart.data.flatMap((row) => [String(row.source ?? ""), String(row.target ?? "")]))).filter(Boolean).slice(0, 10);
  if (!labels.length) return rowsTableMarkup(chart.data);
  const cell = 44;
  const pad = 90;
  const size = pad + labels.length * cell + 16;
  const valueFor = (source: string, target: string) => numberValue(chart.data.find((row) => row.source === source && row.target === target)?.value);
  const body = labels
    .flatMap((source, yIndex) =>
      labels.map((target, xIndex) => {
        const value = valueFor(source, target);
        const intensity = Math.min(Math.abs(value), 1);
        const color = value >= 0 ? `rgba(15,118,110,${0.12 + intensity * 0.78})` : `rgba(220,38,38,${0.12 + intensity * 0.78})`;
        return `<rect x="${pad + xIndex * cell}" y="${pad + yIndex * cell}" width="${cell - 2}" height="${cell - 2}" fill="${color}"/>`;
      }),
    )
    .join("");
  return svgMarkup(size, size, body);
}

function rowsTableMarkup(rows: Record<string, unknown>[]) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0] ?? {}).slice(0, 8);
  return `<table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${rows
    .slice(0, 60)
    .map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function svgMarkup(width: number, height: number, body: string) {
  return `<svg viewBox="0 0 ${width} ${height}" style="width:100%;max-height:360px;background:#f8fafc;border-radius:8px;margin:12px 0">${body}</svg>`;
}

function escapeHtml(value: string) {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function numberValue(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "string") return Number(value.replace(/,/g, ""));
  return Number.NaN;
}

function shortLabel(value: string) {
  return value.length > 10 ? `${value.slice(0, 9)}…` : value;
}

export function valueText(value: number | string | null | undefined) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
}

function arcPath(cx: number, cy: number, radius: number, start: number, end: number) {
  const startX = cx + radius * Math.cos(start);
  const startY = cy + radius * Math.sin(start);
  const endX = cx + radius * Math.cos(end);
  const endY = cy + radius * Math.sin(end);
  const largeArc = end - start > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} 1 ${endX} ${endY} Z`;
}

function quartiles(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    min: sorted[0] ?? 0,
    q1: percentile(sorted, 0.25),
    median: percentile(sorted, 0.5),
    q3: percentile(sorted, 0.75),
    max: sorted[sorted.length - 1] ?? 0,
  };
}

function percentile(sorted: number[], p: number) {
  if (!sorted.length) return 0;
  const index = (sorted.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? sorted[lower] ?? 0) * weight;
}

export function DownloadButton({ label, fileName, content, mime }: { label: string; fileName: string; content: string; mime: string }) {
  const href = URL.createObjectURL(new Blob([content], { type: `${mime};charset=utf-8` }));
  return (
    <a href={href} download={fileName} className="small-button inline-flex">
      {label}
    </a>
  );
}


function multimodalKindLabel(kind: MultimodalInput["kind"]) {
  const labels: Record<MultimodalInput["kind"], string> = {
    image: "图片",
    chart: "图表",
    pdf_page: "PDF/文档页",
    screenshot: "截图",
    note: "文本备注",
  };
  return labels[kind] ?? kind;
}

function multimodalStatusLabel(item: MultimodalInput) {
  const status = item.processing_status;
  if (status === "native_image_payload") return "已进入 Kimi 视觉上下文";
  if (status === "pdf_text_extracted") return "PDF 文本已抽取";
  if (status === "pdf_text_unavailable") return "PDF 文本未抽取";
  if (status === "text_context") return "已作为文本上下文";
  if (item.data_url && item.media_type?.startsWith("image/")) return "已进入 Kimi 视觉上下文";
  if (item.kind === "pdf_page") return "PDF 待抽取/已降级";
  return "已作为上下文";
}

function multimodalStatusClass(status?: string | null) {
  if (status === "native_image_payload" || status === "pdf_text_extracted") {
    return "bg-emerald-50 text-emerald-700";
  }
  if (status === "pdf_text_unavailable") {
    return "bg-amber-50 text-amber-700";
  }
  return "bg-blue-50 text-blue-700";
}
