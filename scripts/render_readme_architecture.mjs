import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = resolve(root, "docs", "assets");
const width = 1400;
const height = 870;
const workflowHeight = 716;

const locales = {
  en: {
    output: "datamind-architecture-en.svg",
    layers: [
      "Product experience",
      "API and control plane",
      "LangGraph data-agent runtime",
      "Durable services and execution boundaries",
    ],
    experience: [
      ["Data workspace", "Upload · Clean · Relate"],
      ["Analysis tasks", "Workflow · History · Trace"],
      ["Reports", "Evidence · Versions · Export"],
      ["Kimi assistant", "Ask · Execute · Attach"],
    ],
    control: [
      ["FastAPI API", "Typed HTTP contracts"],
      ["Session security", "Cookie · CSRF · Limits"],
      ["Permission service", "Scope · Grants · Audit"],
      ["Task control", "SSE · Cancel · Retry · Resume"],
    ],
    agents: [
      ["Cleaning Loop", "Decide · Execute", "Gate · Commit"],
      ["Intent Guard", "Compile · Validate", "Repair · Confirm"],
      ["Planner", "Profile · Semantics", "Analysis contract"],
      ["Analysis Loop", "Safe SQL · Python", "Repair · Charts"],
      ["Verifier", "Grain · Statistics", "Evidence"],
      ["Reviewer", "Adversarial", "validation"],
      ["Report Loop", "Draft · Repair", "Verify · Commit"],
      ["Assistant Graph", "Retrieve · Tools", "Confirm · Answer"],
    ],
    infrastructure: [
      ["Data store", "PostgreSQL · SQLite", "Assets · Versions · Reports"],
      ["LangMem Store", "BaseStore adapter", "Versions · Audit · Guard"],
      ["BGE embedding", "Chinese semantics", "User-scoped cache"],
      ["Python Runner", "Disposable container", "No network"],
      ["Checkpoints", "Resume · Idempotency", "Ordered events"],
      ["Tool runtime", "Schema · Policy", "Invocation"],
      ["Redis + Celery", "Broker · Worker", "Beat"],
    ],
    modelBoundary: {
      label: "Model execution boundary",
      steps: ["Node Harness", "Context Budget", "Model Router", "LLM Provider"],
    },
    flowLabels: [
      "HTTPS",
      "dispatch",
      "repair / replan",
      "Shared runtime services · calls · state · queue · events",
    ],
  },
  zh: {
    output: "datamind-architecture-zh.svg",
    layers: ["产品交互层", "API 与控制平面", "LangGraph 数据智能体运行时", "持久服务与执行边界"],
    experience: [
      ["数据工作台", "上传 · 清洗 · 关系"],
      ["分析任务", "Workflow · 历史 · Trace"],
      ["分析报告", "证据 · 版本 · 导出"],
      ["Kimi 助手", "问答 · 执行 · 附件"],
    ],
    control: [
      ["FastAPI API", "类型化 HTTP 契约"],
      ["会话安全", "Cookie · CSRF · 限流"],
      ["权限服务", "范围 · 授权 · 审计"],
      ["任务控制", "SSE · 取消 · 重试 · 恢复"],
    ],
    agents: [
      ["清洗 Loop", "决策 · 执行", "门禁 · 提交"],
      ["意图 Guard", "编译 · 校验", "修复 · 确认"],
      ["Planner", "画像 · 语义计划", "分析契约"],
      ["分析 Loop", "安全 SQL · Python", "修复 · 图表"],
      ["验证器", "粒度 · 统计", "证据"],
      ["Reviewer", "对抗", "审查"],
      ["报告 Loop", "草稿 · 修复", "验证 · 提交"],
      ["Assistant Graph", "检索 · 工具", "确认 · 回答"],
    ],
    infrastructure: [
      ["数据存储", "PostgreSQL · SQLite", "资产 · 版本 · 报告"],
      ["LangMem Store", "BaseStore Adapter", "版本 · 审计 · Guard"],
      ["BGE Embedding", "中文语义排序", "用户隔离缓存"],
      ["Python Runner", "一次性容器", "禁止网络"],
      ["Checkpoint", "恢复 · 幂等", "有序事件"],
      ["工具运行时", "Schema · 策略", "调用控制"],
      ["Redis + Celery", "Broker · Worker", "Beat"],
    ],
    modelBoundary: {
      label: "模型执行边界",
      steps: ["节点 Harness", "上下文预算", "模型路由", "LLM Provider"],
    },
    flowLabels: [
      "HTTPS",
      "任务调度",
      "修复 / 重规划",
      "共享运行支撑 · 调用 · 状态 · 队列 · 事件",
    ],
  },
};

const workflowLocales = {
  en: {
    output: "datamind-workflow-en.svg",
    title: "DataMind end-to-end workflow",
    description:
      "Data preparation, bounded autonomous analysis, verification, reporting, and Kimi follow-up.",
    layers: [
      "1 · Prepare trusted data",
      "2 · Analyze with bounded loops",
      "3 · Deliver evidence-backed answers",
    ],
    preparation: [
      ["Import files", "Batch · Drag and drop", "Disk-backed staging", "experience"],
      ["Cleaning Loop", "Rules · LLM · Hybrid", "Repair · Quality gate", "loop"],
      ["Validated versions", "Diff · Activate · Rollback", "No unsafe overwrite", "verify"],
      ["Profile & semantics", "Drift · Roles · Metrics", "Relationship graph", "control"],
    ],
    analysis: [
      ["Intent Guard", "Compile · Validate", "Repair · Confirm", "verify"],
      ["Planner", "Question · Scope · Grain", "Analysis contract", "loop"],
      ["Tool Loop", "Safe SQL · Python", "Evidence artifacts", "loop"],
      ["Verifier", "Join grain · Statistics", "Numeric evidence", "verify"],
      ["Reviewer", "Adversarial checks", "Replan on failure", "verify"],
      ["Report Loop", "Draft · Repair · Verify", "Idempotent commit", "report"],
    ],
    delivery: [
      ["Report artifact", "Charts · Lineage · Versions", "HTML · Markdown · PDF", "report"],
      ["Kimi assistant", "Evidence · Scoped memory", "Ask · Execute · Attach", "experience"],
      ["Next action", "Follow-up · Reanalyze", "Audited and recoverable", "control"],
    ],
    labels: [
      "bounded repair / replan",
      "Durable jobs · checkpoints · ordered SSE events",
    ],
    modelGuard:
      "Every model call · Node Harness → budget evaluation → deterministic reduction → Router gate",
  },
  zh: {
    output: "datamind-workflow-zh.svg",
    title: "DataMind 端到端流程",
    description: "从数据准备到自主分析、验证、报告和 Kimi 后续操作的完整流程。",
    layers: ["1 · 准备可信数据", "2 · 有边界的自主分析", "3 · 交付证据化答案"],
    preparation: [
      ["导入文件", "批量 · 拖拽上传", "大文件落盘暂存", "experience"],
      ["清洗 Loop", "规则 · LLM · Hybrid", "修复 · 质量门禁", "loop"],
      ["可信清洗版本", "Diff · 激活 · 回滚", "失败不覆盖当前版本", "verify"],
      ["画像与语义", "漂移 · 角色 · 指标", "实体关系图", "control"],
    ],
    analysis: [
      ["意图 Guard", "编译 · 极性校验", "修复 · 用户确认", "verify"],
      ["Planner", "问题 · 范围 · 粒度", "分析契约", "loop"],
      ["工具 Loop", "安全 SQL · Python", "证据 Artifact", "loop"],
      ["验证器", "Join 粒度 · 统计", "数值证据", "verify"],
      ["Reviewer", "对抗审查", "失败触发重规划", "verify"],
      ["报告 Loop", "草稿 · 修复 · 验证", "幂等提交", "report"],
    ],
    delivery: [
      ["报告资产", "图表 · 血缘 · 版本", "HTML · Markdown · PDF", "report"],
      ["Kimi 助手", "证据 · 范围记忆", "问答 · 执行 · 附件", "experience"],
      ["后续行动", "追问 · 重新分析", "可审计 · 可恢复", "control"],
    ],
    labels: ["有边界的修复 / 重规划", "持久任务 · Checkpoint · 有序 SSE 事件"],
    modelGuard: "所有模型调用 · 节点 Harness → 预算评估 → 确定性压缩 → Router 门禁",
  },
};

const tones = {
  experience: ["#ecfdf5", "#10b981", "#064e3b"],
  control: ["#eff6ff", "#60a5fa", "#1e3a8a"],
  loop: ["#f0fdf4", "#22c55e", "#14532d"],
  verify: ["#fffbeb", "#f59e0b", "#78350f"],
  report: ["#faf5ff", "#c084fc", "#581c87"],
  infrastructure: ["#f8fafc", "#94a3b8", "#334155"],
  runner: ["#fff7ed", "#fb923c", "#7c2d12"],
};

const escapeXml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

function band(label, y, bandHeight, fill) {
  return `
    <rect x="16" y="${y}" width="1368" height="${bandHeight}" rx="8" fill="${fill}" stroke="#dbe4ef"/>
    <text x="34" y="${y + 27}" class="layer">${escapeXml(label)}</text>`;
}

function card({ x, y, w, h, title, details, tone, titleSize = 20 }) {
  const [fill, stroke, text] = tones[tone];
  const firstDetailY = y + (details.length > 1 ? 66 : 70);
  const detailText = details
    .map(
      (line, index) =>
        `<tspan x="${x + w / 2}" y="${firstDetailY + index * 21}">${escapeXml(line)}</tspan>`,
    )
    .join("");
  return `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="2"/>
    <text x="${x + w / 2}" y="${y + 36}" text-anchor="middle" class="card-title" font-size="${titleSize}" fill="${text}">${escapeXml(title)}</text>
    <text text-anchor="middle" class="card-detail" fill="${text}">${detailText}</text>`;
}

function arrow(x1, y1, x2, y2, { dashed = false, muted = false } = {}) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="arrow${muted ? " muted" : ""}"${dashed ? ' stroke-dasharray="7 7"' : ""} marker-end="url(#arrow)"/>`;
}

function sharedConnection(x1, y1, x2, y2) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="shared-connection" marker-start="url(#arrow)" marker-end="url(#arrow)"/>`;
}

function serviceLink(x1, y1, x2, y2) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="shared-connection"/>`;
}

function label(text, x, y) {
  return `<text x="${x}" y="${y}" text-anchor="middle" class="flow-label">${escapeXml(text)}</text>`;
}

function svgStart({ svgHeight, title, description }) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${svgHeight}" viewBox="0 0 ${width} ${svgHeight}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(title)}</title>
  <desc id="desc">${escapeXml(description)}</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse" markerUnits="strokeWidth">
      <path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/>
    </marker>
    <style>
      text { font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif; }
      .layer { font-size: 18px; font-weight: 700; fill: #0f172a; }
      .card-title { font-weight: 700; }
      .card-detail { font-size: 15px; font-weight: 500; }
      .arrow { stroke: #64748b; stroke-width: 2; fill: none; }
      .arrow.muted { stroke: #94a3b8; stroke-width: 1.7; }
      .shared-connection { stroke: #64748b; stroke-width: 1.7; fill: none; }
      .flow-label { font-size: 14px; font-weight: 700; fill: #475569; }
      .service-bus { font-size: 14px; font-weight: 700; fill: #334155; }
    </style>
  </defs>
  <rect width="${width}" height="${svgHeight}" fill="#ffffff"/>`;
}

function render(locale) {
  const experienceX = [63, 388, 713, 1038];
  const controlX = [...experienceX];
  const agentX = Array.from({ length: 8 }, (_, index) => 18 + index * 171);
  const infrastructureX = Array.from(
    { length: locale.infrastructure.length },
    (_, index) => 20 + index * 196,
  );
  const agentTones = [
    "loop",
    "verify",
    "loop",
    "loop",
    "verify",
    "verify",
    "report",
    "loop",
  ];
  const infrastructureTones = [
    "infrastructure",
    "infrastructure",
    "infrastructure",
    "runner",
    "infrastructure",
    "infrastructure",
    "infrastructure",
  ];

  const parts = [
    svgStart({
      svgHeight: height,
      title: "DataMind architecture",
      description:
        "DataMind product, API, data-agent workflow, and infrastructure architecture.",
    }),
  ];

  parts.push(band(locale.layers[0], 16, 150, "#f8fffc"));
  parts.push(band(locale.layers[1], 181, 140, "#f8fbff"));
  parts.push(band(locale.layers[2], 336, 220, "#fbfdfc"));
  parts.push(band(locale.layers[3], 571, 283, "#fafbfc"));

  locale.experience.forEach(([title, detail], index) => {
    parts.push(
      card({
        x: experienceX[index],
        y: 66,
        w: 290,
        h: 76,
        title,
        details: [detail],
        tone: "experience",
      }),
    );
  });

  locale.control.forEach(([title, detail], index) => {
    parts.push(
      card({
        x: controlX[index],
        y: 221,
        w: 290,
        h: 76,
        title,
        details: [detail],
        tone: "control",
      }),
    );
  });

  locale.agents.forEach(([title, ...details], index) => {
    parts.push(
      card({
        x: agentX[index],
        y: 386,
        w: 157,
        h: 120,
        title,
        details,
        tone: agentTones[index],
        titleSize: 16,
      }),
    );
  });

  locale.infrastructure.forEach(([title, ...details], index) => {
    parts.push(
      card({
        x: infrastructureX[index],
        y: 714,
        w: 184,
        h: 128,
        title,
        details,
        tone: infrastructureTones[index],
        titleSize: 15,
      }),
    );
  });

  parts.push(arrow(700, 166, 700, 221));
  parts.push(label(locale.flowLabels[0], 735, 198));
  parts.push(arrow(700, 321, 700, 386));
  parts.push(label(locale.flowLabels[1], 750, 357));

  for (let index = 0; index < agentX.length - 1; index += 1) {
    parts.push(arrow(agentX[index] + 157, 446, agentX[index + 1], 446));
  }

  const agentCenters = agentX.map((x) => x + 78.5);
  parts.push(sharedConnection(700, 556, 700, 610));
  parts.push(
    `<rect x="226" y="610" width="948" height="46" rx="8" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>`,
  );
  parts.push(
    `<text x="244" y="638" class="service-bus">${escapeXml(locale.modelBoundary.label)}</text>`,
  );
  const boundaryX = [452, 638, 824, 1010];
  locale.modelBoundary.steps.forEach((step, index) => {
    parts.push(
      `<rect x="${boundaryX[index]}" y="618" width="148" height="30" rx="7" fill="#eff6ff" stroke="#60a5fa"/>`,
    );
    parts.push(
      `<text x="${boundaryX[index] + 74}" y="638" text-anchor="middle" class="service-bus" fill="#1e3a8a">${escapeXml(step)}</text>`,
    );
    if (index < boundaryX.length - 1) {
      parts.push(arrow(boundaryX[index] + 148, 633, boundaryX[index + 1], 633));
    }
  });
  parts.push(
    `<rect x="226" y="670" width="948" height="30" rx="8" fill="#eef2f7" stroke="#94a3b8"/>`,
  );
  parts.push(
    `<text x="700" y="690" text-anchor="middle" class="service-bus">${escapeXml(locale.flowLabels[3])}</text>`,
  );
  parts.push(serviceLink(700, 656, 700, 670));
  const infrastructureCenters = infrastructureX.map((x) => x + 92);
  infrastructureCenters.forEach((x) => parts.push(serviceLink(x, 700, x, 714)));

  const reportCenter = agentCenters[6];
  const plannerCenter = agentCenters[2];
  parts.push(
    `<path d="M ${reportCenter} 506 C ${reportCenter} 548, ${plannerCenter} 548, ${plannerCenter} 506" class="arrow muted" stroke-dasharray="7 7" marker-end="url(#arrow)"/>`,
  );
  parts.push(label(locale.flowLabels[2], (reportCenter + plannerCenter) / 2, 542));
  parts.push(`</svg>`);
  return parts.join("\n");
}

function renderWorkflow(locale) {
  const preparationX = [63, 388, 713, 1038];
  const analysisX = [22, 250, 478, 706, 934, 1162];
  const deliveryX = [200, 550, 900];
  const parts = [
    svgStart({
      svgHeight: workflowHeight,
      title: locale.title,
      description: locale.description,
    }),
  ];

  parts.push(band(locale.layers[0], 16, 180, "#f8fffc"));
  parts.push(
    `<rect x="230" y="204" width="940" height="28" rx="8" fill="#eff6ff" stroke="#60a5fa"/>`,
  );
  parts.push(
    `<text x="700" y="223" text-anchor="middle" class="service-bus" fill="#1e3a8a">${escapeXml(locale.modelGuard)}</text>`,
  );
  parts.push(band(locale.layers[1], 242, 260, "#fbfdfc"));
  parts.push(band(locale.layers[2], 517, 183, "#fafbff"));

  locale.preparation.forEach(([title, ...rest], index) => {
    const tone = rest.pop();
    parts.push(
      card({
        x: preparationX[index],
        y: 66,
        w: 290,
        h: 105,
        title,
        details: rest,
        tone,
        titleSize: 19,
      }),
    );
  });

  locale.analysis.forEach(([title, ...rest], index) => {
    const tone = rest.pop();
    parts.push(
      card({
        x: analysisX[index],
        y: 292,
        w: 216,
        h: 130,
        title,
        details: rest,
        tone,
        titleSize: 18,
      }),
    );
  });

  locale.delivery.forEach(([title, ...rest], index) => {
    const tone = rest.pop();
    parts.push(
      card({
        x: deliveryX[index],
        y: 567,
        w: 300,
        h: 110,
        title,
        details: rest,
        tone,
        titleSize: 19,
      }),
    );
  });

  for (let index = 0; index < preparationX.length - 1; index += 1) {
    parts.push(arrow(preparationX[index] + 290, 118, preparationX[index + 1], 118));
  }
  parts.push(
    `<path d="M 1183 171 V 236 H 160 V 292" class="arrow" marker-end="url(#arrow)"/>`,
  );

  for (let index = 0; index < analysisX.length - 1; index += 1) {
    parts.push(arrow(analysisX[index] + 216, 357, analysisX[index + 1], 357));
  }
  parts.push(
    `<path d="M 1270 422 C 1270 456, 130 456, 130 422" class="arrow muted" stroke-dasharray="7 7" marker-end="url(#arrow)"/>`,
  );
  parts.push(label(locale.labels[0], 700, 449));
  parts.push(
    `<rect x="230" y="470" width="940" height="24" rx="8" fill="#eef2f7" stroke="#94a3b8"/>`,
  );
  parts.push(
    `<text x="700" y="487" text-anchor="middle" class="service-bus">${escapeXml(locale.labels[1])}</text>`,
  );
  parts.push(
    `<path d="M 1360 357 H 1372 V 510 H 350 V 567" class="arrow" marker-end="url(#arrow)"/>`,
  );

  for (let index = 0; index < deliveryX.length - 1; index += 1) {
    parts.push(arrow(deliveryX[index] + 300, 622, deliveryX[index + 1], 622));
  }

  parts.push(`</svg>`);
  return parts.join("\n");
}

mkdirSync(outputDir, { recursive: true });
const generated = [];
for (const [language, locale] of Object.entries(locales)) {
  const svgPath = resolve(outputDir, locale.output);
  writeFileSync(svgPath, render(locale), "utf8");
  generated.push({
    svgPath,
    pngPath: svgPath.replace(/\.svg$/, ".png"),
    svgHeight: height,
  });

  const workflowLocale = workflowLocales[language];
  const workflowSvgPath = resolve(outputDir, workflowLocale.output);
  writeFileSync(workflowSvgPath, renderWorkflow(workflowLocale), "utf8");
  generated.push({
    svgPath: workflowSvgPath,
    pngPath: workflowSvgPath.replace(/\.svg$/, ".png"),
    svgHeight: workflowHeight,
  });
}

const requireFromFrontend = createRequire(resolve(root, "frontend", "react", "package.json"));
const { chromium } = requireFromFrontend("@playwright/test");
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width, height } });
  for (const { svgPath, pngPath, svgHeight } of generated) {
    await page.setViewportSize({ width, height: svgHeight });
    await page.goto(pathToFileURL(svgPath).href);
    await page.screenshot({ path: pngPath });
  }
} finally {
  await browser.close();
}
