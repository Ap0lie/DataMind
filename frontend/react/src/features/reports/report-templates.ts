export type ReportTemplateMode = "brief" | "standard" | "detailed";

type TemplateChart = { title: string; explanation?: string };
type TemplateFinding = { title: string; content: string };
type TemplateValidationIssue = { severity: string; finding_ref: string; issue: string };

type TemplateReport = {
  executive_summary: string;
  key_findings?: TemplateFinding[];
  charts?: TemplateChart[];
  chart_explanations?: string[];
  recommended_next_steps?: string[];
  data_gaps?: string[];
  validation_issues?: TemplateValidationIssue[];
  sql_results?: Record<string, unknown>[];
  analysis_trace?: unknown[];
};

export function reportTemplatePrompt(mode: ReportTemplateMode) {
  if (mode === "brief") {
    return "生成面向管理者的简报：摘要不超过两段，最多 3 条核心发现、4 张必要图表和 3 条行动建议；隐藏技术轨迹，但保留验证风险。";
  }
  if (mode === "detailed") {
    return "生成详细分析报告：保留分析上下文、完整证据、验证问题、SQL 摘要、数据缺口和分析轨迹，结构清晰且避免重复。";
  }
  return "生成标准分析报告：平衡业务结论、关键证据、图表和行动建议；技术细节保持精炼。";
}

export function reportChartPrompt(mode: ReportTemplateMode) {
  const limit = mode === "brief" ? 4 : mode === "standard" ? 8 : 10;
  return `图表服务于结论，最多 ${limit} 张；标题采用业务语言，颜色有区分度，按业务意义排序，并使用聚合或分箱数据。`;
}

export function reportTemplateLabel(mode: ReportTemplateMode) {
  return { brief: "简报", standard: "标准", detailed: "详细" }[mode];
}

export function reportForTemplate<T extends TemplateReport>(report: T, mode: ReportTemplateMode): T {
  if (mode === "detailed") return report;
  const brief = mode === "brief";
  const charts = (report.charts ?? []).slice(0, brief ? 4 : 8);
  return {
    ...report,
    key_findings: (report.key_findings ?? []).slice(0, brief ? 3 : 6),
    charts,
    chart_explanations: charts.map((chart) => chart.explanation ?? "").filter(Boolean),
    recommended_next_steps: (report.recommended_next_steps ?? []).slice(0, brief ? 3 : 5),
    data_gaps: (report.data_gaps ?? []).slice(0, brief ? 2 : 5),
    validation_issues: (report.validation_issues ?? []).slice(0, brief ? 3 : 8),
    sql_results: brief ? [] : (report.sql_results ?? []).slice(0, 20),
    analysis_trace: [],
  } as T;
}

export function markdownReportForTemplate(title: string, report: TemplateReport) {
  const lines = [`# ${title}`, "", "## 核心摘要", report.executive_summary, ""];
  if (report.key_findings?.length) {
    lines.push(
      "## 核心发现",
      ...report.key_findings.map((item) => `- **${item.title}**：${item.content}`),
      "",
    );
  }
  if (report.charts?.length) {
    lines.push(
      "## 图表",
      ...report.charts.map((item) => `- **${item.title}**：${item.explanation || "见渲染报告。"}`),
      "",
    );
  }
  if (report.validation_issues?.length) {
    lines.push(
      "## 校验提示",
      ...report.validation_issues.map(
        (item) => `- ${item.severity} · ${item.finding_ref}：${item.issue}`,
      ),
      "",
    );
  }
  if (report.recommended_next_steps?.length) {
    lines.push(
      "## 下一步建议",
      ...report.recommended_next_steps.map((item) => `- ${item}`),
      "",
    );
  }
  return lines.join("\n");
}
