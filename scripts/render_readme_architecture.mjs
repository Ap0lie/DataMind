import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = resolve(root, "docs", "assets");
const width = 1400;
const height = 795;

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
      ["Planner", "Profile · Semantics", "Analysis contract"],
      ["Analysis Loop", "Safe SQL · Python", "Repair · Charts"],
      ["Verifier", "Grain · Statistics", "Evidence"],
      ["Reviewer", "Adversarial", "validation"],
      ["Report Loop", "Draft · Repair", "Verify · Commit"],
      ["Assistant Graph", "Retrieve · Tools", "Confirm · Answer"],
    ],
    infrastructure: [
      ["Data store", "PostgreSQL · SQLite", "Assets · Versions · Reports"],
      ["BGE embedding", "Chinese semantics", "User-scoped cache"],
      ["Python Runner", "Disposable container", "No network"],
      ["Checkpoints", "Resume · Idempotency", "Ordered events"],
      ["Tool runtime", "Schema · Policy", "Invocation"],
      ["LLM providers", "DeepSeek · Kimi", "Role routing"],
      ["Redis + Celery", "Broker · Worker", "Beat"],
    ],
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
      ["Planner", "画像 · 语义计划", "分析契约"],
      ["分析 Loop", "安全 SQL · Python", "修复 · 图表"],
      ["验证器", "粒度 · 统计", "证据"],
      ["Reviewer", "对抗", "审查"],
      ["报告 Loop", "草稿 · 修复", "验证 · 提交"],
      ["Assistant Graph", "检索 · 工具", "确认 · 回答"],
    ],
    infrastructure: [
      ["数据存储", "PostgreSQL · SQLite", "资产 · 版本 · 报告"],
      ["BGE Embedding", "中文语义排序", "用户隔离缓存"],
      ["Python Runner", "一次性容器", "禁止网络"],
      ["Checkpoint", "恢复 · 幂等", "有序事件"],
      ["工具运行时", "Schema · 策略", "调用控制"],
      ["LLM Provider", "DeepSeek · Kimi", "按角色路由"],
      ["Redis + Celery", "Broker · Worker", "Beat"],
    ],
    flowLabels: [
      "HTTPS",
      "任务调度",
      "修复 / 重规划",
      "共享运行支撑 · 调用 · 状态 · 队列 · 事件",
    ],
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

function render(locale) {
  const experienceX = [63, 388, 713, 1038];
  const controlX = [...experienceX];
  const agentX = Array.from({ length: 7 }, (_, index) => 22 + index * 196);
  const agentTones = ["loop", "loop", "loop", "verify", "verify", "report", "loop"];
  const infrastructureTones = [
    "infrastructure",
    "infrastructure",
    "runner",
    "infrastructure",
    "infrastructure",
    "infrastructure",
    "infrastructure",
  ];

  const parts = [`<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">DataMind architecture</title>
  <desc id="desc">DataMind product, API, data-agent workflow, and infrastructure architecture.</desc>
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
  <rect width="${width}" height="${height}" fill="#ffffff"/>`];

  parts.push(band(locale.layers[0], 16, 150, "#f8fffc"));
  parts.push(band(locale.layers[1], 181, 140, "#f8fbff"));
  parts.push(band(locale.layers[2], 336, 220, "#fbfdfc"));
  parts.push(band(locale.layers[3], 571, 208, "#fafbfc"));

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
        w: 180,
        h: 120,
        title,
        details,
        tone: agentTones[index],
        titleSize: 17,
      }),
    );
  });

  locale.infrastructure.forEach(([title, ...details], index) => {
    parts.push(
      card({
        x: agentX[index],
        y: 648,
        w: 180,
        h: 124,
        title,
        details,
        tone: infrastructureTones[index],
        titleSize: 17,
      }),
    );
  });

  parts.push(arrow(700, 166, 700, 221));
  parts.push(label(locale.flowLabels[0], 735, 198));
  parts.push(arrow(700, 321, 700, 386));
  parts.push(label(locale.flowLabels[1], 750, 357));

  for (let index = 0; index < agentX.length - 1; index += 1) {
    parts.push(arrow(agentX[index] + 180, 446, agentX[index + 1], 446));
  }

  const agentCenters = agentX.map((x) => x + 90);
  parts.push(
    `<rect x="226" y="604" width="948" height="30" rx="8" fill="#eef2f7" stroke="#94a3b8"/>`,
  );
  parts.push(
    `<text x="700" y="624" text-anchor="middle" class="service-bus">${escapeXml(locale.flowLabels[3])}</text>`,
  );
  parts.push(sharedConnection(700, 556, 700, 604));
  agentCenters.forEach((x) => parts.push(serviceLink(x, 634, x, 648)));

  const reportCenter = agentCenters[5];
  const plannerCenter = agentCenters[1];
  parts.push(
    `<path d="M ${reportCenter} 506 C ${reportCenter} 548, ${plannerCenter} 548, ${plannerCenter} 506" class="arrow muted" stroke-dasharray="7 7" marker-end="url(#arrow)"/>`,
  );
  parts.push(label(locale.flowLabels[2], (reportCenter + plannerCenter) / 2, 542));
  parts.push(`</svg>`);
  return parts.join("\n");
}

mkdirSync(outputDir, { recursive: true });
const generated = [];
for (const locale of Object.values(locales)) {
  const svgPath = resolve(outputDir, locale.output);
  writeFileSync(svgPath, render(locale), "utf8");
  generated.push({ svgPath, pngPath: svgPath.replace(/\.svg$/, ".png") });
}

const requireFromFrontend = createRequire(resolve(root, "frontend", "react", "package.json"));
const { chromium } = requireFromFrontend("@playwright/test");
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width, height } });
  for (const { svgPath, pngPath } of generated) {
    await page.goto(pathToFileURL(svgPath).href);
    await page.screenshot({ path: pngPath });
  }
} finally {
  await browser.close();
}
