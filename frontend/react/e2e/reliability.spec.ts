import os from "node:os";
import path from "node:path";
import { expect, test } from "@playwright/test";

function cleaningJob(datasetId: string, jobId: string, status: "running" | "completed") {
  const completed = status === "completed";
  return {
    job_id: jobId,
    dataset_id: datasetId,
    requirement: "",
    cleaning_strategy: "auto",
    selected_strategy: completed ? "rules" : null,
    status,
    progress: completed ? 100 : 35,
    current_stage: completed ? "complete" : "cleaning_execute",
    events: completed
      ? [{ sequence: 2, stage: "cleaning_commit", status: "completed", progress: 100, message: "Validated cleaning version committed and activated.", event_type: "cleaning_commit", iteration: 1, strategy: "rules", payload: {}, created_at: "2026-07-11T09:00:02Z" }]
      : [{ sequence: 1, stage: "cleaning_execute", status: "completed", progress: 35, message: "Cleaning strategy rules executed in isolation.", event_type: "cleaning_execution", iteration: 1, strategy: "rules", payload: {}, created_at: "2026-07-11T09:00:01Z" }],
    loop_summary: {},
    terminal_reason: completed ? "validated" : null,
    error: null,
    cleaning_run_id: completed ? "77777777-7777-4777-8777-777777777777" : null,
    last_event_sequence: completed ? 2 : 1,
  };
}

test("login, workflow navigation, and logout remain operational", async ({ page }, testInfo) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page).toHaveTitle(/DataMind/i);
  await expect(page.getByRole("heading", { name: "DataMind" })).toBeVisible();

  await page.getByLabel("用户名").fill(`qa_${testInfo.project.name.replace(/\W/g, "_")}`);
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();

  await expect(page.getByRole("navigation")).toBeVisible();
  await expect(page.getByRole("heading", { name: "工作区" })).toBeVisible();
  await expect(page.getByRole("button", { name: /导入数据/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /开始分析/ })).toBeVisible();
  await page.screenshot({
    path: path.join(os.tmpdir(), `datamind-dashboard-${testInfo.project.name}.png`),
    fullPage: false,
  });
  await page.getByRole("button", { name: /导入数据/ }).click();
  await expect(page.getByRole("heading", { name: "导入工作台" })).toBeVisible();
  await expect(page.getByRole("tab", { name: /资产列表/ })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("tab", { name: /关系管理/ }).click();
  await expect(page.getByRole("tab", { name: /关系管理/ })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("tab", { name: /资产列表/ }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(os.tmpdir(), `datamind-datasets-${testInfo.project.name}.png`),
    fullPage: false,
  });
  await page.getByRole("button", { name: "分析任务" }).click();
  await expect(page.getByRole("heading", { name: "分析记录" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "新建分析" })).toBeVisible();
  await expect(page.getByLabel("分析问题")).toBeVisible();
  await page.screenshot({
    path: path.join(os.tmpdir(), `datamind-${testInfo.project.name}.png`),
    fullPage: false,
  });

  if (testInfo.project.name.includes("desktop")) await page.setViewportSize({ width: 1912, height: 949 });
  await page.getByRole("button", { name: "Kimi" }).click();
  await expect(page.getByRole("heading", { name: "数据分析助手" })).toBeVisible();
  if (testInfo.project.name.includes("desktop")) {
    const workspaceGaps = await page.locator(".assistant-workspace").evaluate((workspace) => {
      const main = workspace.closest("main");
      if (!main) return { left: 999, right: 999, top: 999, bottom: 999 };
      const workspaceBox = workspace.getBoundingClientRect();
      const mainBox = main.getBoundingClientRect();
      return {
        left: Math.abs(workspaceBox.left - mainBox.left),
        right: Math.abs(mainBox.right - workspaceBox.right),
        top: Math.abs(workspaceBox.top - mainBox.top),
        bottom: Math.abs(mainBox.bottom - workspaceBox.bottom),
      };
    });
    expect(Math.max(...Object.values(workspaceGaps))).toBeLessThanOrEqual(1);
  }
  await expect(page.getByText("问答：仅读取与计划预览")).toBeVisible();
  await page.getByRole("button", { name: "执行任务" }).click();
  await expect(page.getByText("执行任务：当前范围自动获得完整管理权限")).toBeVisible();
  await expect(page.getByText("仅在已授权资产内执行")).toBeVisible();
  await page.getByRole("button", { name: "问答", exact: true }).click();
  await expect(page.getByPlaceholder("向 Kimi 询问你的数据、分析结果或报告...")).toBeVisible();
  const assistantFontSizes = await page.locator(".assistant-shell").evaluate((shell) => {
    const fontSize = (selector: string) => {
      const element = shell.querySelector(selector);
      return element ? Number.parseFloat(window.getComputedStyle(element).fontSize) : 0;
    };
    return {
      composer: fontSize(".assistant-composer textarea"),
      modelTitle: fontSize(".assistant-model h2"),
    };
  });
  expect(assistantFontSizes.composer).toBeGreaterThanOrEqual(16);
  expect(assistantFontSizes.modelTitle).toBeGreaterThanOrEqual(16);
  if (testInfo.project.name.includes("desktop")) {
    expect(assistantFontSizes.composer).toBeGreaterThanOrEqual(17);
    expect(assistantFontSizes.modelTitle).toBeGreaterThanOrEqual(18);
    const widthRatio = await page.locator(".assistant-composer").evaluate((composer) => {
      const thread = composer.closest(".assistant-thread");
      return thread ? composer.getBoundingClientRect().width / thread.getBoundingClientRect().width : 0;
    });
    expect(widthRatio).toBeGreaterThan(0.8);
    await page.getByTitle("收起消息记录").click();
    await expect(page.locator(".assistant-workspace.history-collapsed")).toBeVisible();
    await page.getByTitle("展开消息记录").click();
  }
  await page.getByRole("button", { name: /打开 Kimi 工作台/ }).click();
  const workbench = page.getByRole("complementary", { name: "Kimi 权限与操作" });
  await expect(workbench).toBeVisible();
  await expect(workbench.getByText("当前为自动检索范围")).toBeVisible();
  await expect(workbench.getByText("问答会自动查找相关资料；如需执行任务，请在此选择资产授权，或先在顶部切换到具体范围。")).toBeVisible();
  await expect(page.getByPlaceholder("向 Kimi 询问你的数据、分析结果或报告...")).toBeVisible();
  await expect(page.locator(".assistant-control-backdrop")).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollHeight <= window.innerHeight + 1)).toBe(true);
  await expect.poll(() => page.locator(".assistant-control-content").evaluate((element) => getComputedStyle(element).overflowY)).toBe("auto");
  await page.screenshot({ path: path.join(os.tmpdir(), `datamind-assistant-${testInfo.project.name}.png`), fullPage: false });
  await workbench.getByTitle("关闭").click();
  await expect(workbench).toHaveCount(0);

  if (testInfo.project.name.includes("mobile")) {
    await page.getByTitle("打开消息记录").click();
    await expect(page.locator(".assistant-history.open")).toBeVisible();
    await page.getByRole("button", { name: "关闭消息记录" }).click();
    await expect(page.locator(".assistant-history.open")).toHaveCount(0);
    await page.getByRole("button", { name: /打开 Kimi 工作台/ }).click();
    await expect(page.locator(".assistant-history.open")).toHaveCount(0);
    await expect(page.getByRole("complementary", { name: "Kimi 权限与操作" })).toBeVisible();
    await page.getByRole("complementary", { name: "Kimi 权限与操作" }).getByTitle("关闭").click();
  }

  await page.getByRole("button", { name: /退出|Log Out/i }).click();
  await expect(page.getByRole("heading", { name: "DataMind" })).toBeVisible();
  expect(consoleErrors).toEqual([]);
});

test("invalid credentials show a friendly login message", async ({ page }) => {
  await page.route("**/api/v1/auth/login", async (route) => {
    await route.fulfill({
      status: 401,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Invalid username or password." }),
    });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("existing-user");
  await page.getByLabel("密码").fill("wrong-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();

  await expect(page.getByRole("alert")).toHaveText("用户名或密码不正确，请检查后重试。");
  await expect(page.getByText(/接口错误|Invalid username or password/)).toHaveCount(0);
});

test("a dashboard record restores its complete workflow session", async ({ page }) => {
  const jobId = "11111111-1111-4111-8111-111111111111";
  const completedJob = {
    job_id: jobId,
    dataset_id: "22222222-2222-4222-8222-222222222222",
    question: "季度销售趋势与异常月份",
    status: "completed",
    progress: 100,
    current_stage: "complete",
    report_id: "33333333-3333-4333-8333-333333333333",
    events: [
      { sequence: 1, stage: "planner", progress: 12, message: "Profiling dataset and planning analysis route.", status: "running", created_at: "2026-07-10T08:00:01Z" },
      { sequence: 2, stage: "design_framework", progress: 24, message: "Designing analysis framework.", status: "running", created_at: "2026-07-10T08:00:02Z" },
      { sequence: 3, stage: "sql_agent", progress: 42, message: "Running safe SQL analysis.", status: "running", created_at: "2026-07-10T08:00:03Z" },
      { sequence: 4, stage: "python_agent", progress: 62, message: "Running Python analysis.", status: "running", created_at: "2026-07-10T08:00:04Z" },
      { sequence: 5, stage: "format_charts", progress: 78, message: "Formatting report charts.", status: "running", created_at: "2026-07-10T08:00:05Z" },
      { sequence: 6, stage: "adversarial_validate", progress: 90, message: "Reviewing analysis quality and gaps.", status: "running", created_at: "2026-07-10T08:00:06Z" },
      { sequence: 7, stage: "report_agent", progress: 96, message: "Generating and saving report.", status: "running", created_at: "2026-07-10T08:00:07Z" },
      { sequence: 8, stage: "complete", progress: 100, message: "Analysis job completed.", status: "completed", created_at: "2026-07-10T08:00:08Z" },
    ],
    created_at: "2026-07-10T08:00:00Z",
    updated_at: "2026-07-10T08:00:08Z",
    completed_at: "2026-07-10T08:00:08Z",
  };

  await page.route("**/api/v1/analysis/jobs?*", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [completedJob] }) });
  });
  await page.route(`**/api/v1/analysis/jobs/${jobId}`, async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(completedJob) });
  });
  await page.route(`**/api/v1/analysis/jobs/${jobId}/result`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        question: completedJob.question,
        sql_result: { sql: "SELECT month, SUM(sales) FROM dataset GROUP BY month", rows: [] },
        python_result: null,
        report_markdown: "# 季度销售趋势\n分析已完成。",
        structured_report: null,
        workflow_trace: [],
        multimodal_inputs: [],
        multi_dataset_context: null,
        planner_metadata: null,
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_session_history");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();

  await page.getByRole("button", { name: /季度销售趋势与异常月份/ }).click();
  await expect(page.getByRole("heading", { name: "季度销售趋势与异常月份" })).toBeVisible();
  await expect(page.getByText("规划器详情", { exact: false })).toBeVisible();
  await expect(page.getByText("运行日志", { exact: true })).toBeVisible();
  await expect(page.getByText("生成的 SQL", { exact: true })).toBeVisible();
  await expect(page.getByText("SELECT month, SUM(sales) FROM dataset GROUP BY month", { exact: true })).toBeVisible();
  const overflowing = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>("body *"))
      .map((element) => ({
        className: element.className,
        text: element.innerText?.slice(0, 60),
        right: Math.round(element.getBoundingClientRect().right),
      }))
      .filter((item) => item.right > window.innerWidth + 1),
  );
  expect(overflowing).toEqual([]);
  const clippedWorkflowLabels = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>(".agent-plan-pill, .workflow-node"))
      .filter((element) => element.scrollWidth > element.clientWidth + 1)
      .map((element) => element.innerText),
  );
  expect(clippedWorkflowLabels).toEqual([]);
  await page.screenshot({
    path: path.join(os.tmpdir(), `datamind-session-${test.info().project.name}.png`),
    fullPage: false,
  });
});

test("a running workflow stays visible in a floating progress pill", async ({ page }, testInfo) => {
  const jobId = "44444444-4444-4444-8444-444444444444";
  const runningJob = {
    job_id: jobId,
    dataset_id: "55555555-5555-4555-8555-555555555555",
    question: "分析客户留存率变化",
    status: "running",
    progress: 64,
    current_stage: "python_agent",
    events: [
      { sequence: 1, stage: "planner", progress: 12, message: "Profiling dataset and planning analysis route.", status: "running", created_at: "2026-07-11T08:00:01Z" },
      { sequence: 2, stage: "sql_agent", progress: 42, message: "Running safe SQL analysis.", status: "running", created_at: "2026-07-11T08:00:02Z" },
      { sequence: 3, stage: "python_agent", progress: 64, message: "Running Python analysis.", status: "running", created_at: "2026-07-11T08:00:03Z" },
    ],
    created_at: "2026-07-11T08:00:00Z",
    updated_at: "2026-07-11T08:00:03Z",
  };

  await page.route("**/api/v1/analysis/jobs?*", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [runningJob] }) });
  });
  await page.route(`**/api/v1/analysis/jobs/${jobId}`, async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(runningJob) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_floating_progress");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();

  await page.getByRole("button", { name: "数据集" }).click();
  await expect(page.getByRole("heading", { name: "导入工作台" })).toBeVisible();
  const progressPill = page.getByRole("button", { name: /查看运行中的分析：分析客户留存率变化，64%/ });
  await expect(progressPill).toBeVisible();
  await expect(progressPill).toContainText("Python 智能体");
  await expect(progressPill).toContainText("64%");
  await page.screenshot({
    path: path.join(os.tmpdir(), `datamind-floating-task-${testInfo.project.name}.png`),
    fullPage: false,
  });

  await progressPill.click();
  await expect(page.getByRole("heading", { name: "分析客户留存率变化" })).toBeVisible();
  await expect(progressPill).toHaveCount(0);
});

test("dataset cleaning progress survives navigation", async ({ page }) => {
  const datasetId = "66666666-6666-4666-8666-666666666666";
  let releaseCleaning!: () => void;
  let cleaningReleased = false;
  const cleaningGate = new Promise<void>((resolve) => {
    releaseCleaning = () => {
      cleaningReleased = true;
      resolve();
    };
  });

  await page.route("**/api/v1/store/files/import", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        dataset: {
          dataset_id: datasetId,
          name: "navigation-cleaning.csv",
          source_type: "csv",
          status: "imported",
          source_metadata: {},
          created_at: "2026-07-11T09:00:00Z",
        },
        inserted: 2,
        preview_records: [{ customer_id: 1, value: 10 }, { customer_id: 2, value: 20 }],
      }),
    });
  });
  const cleaningJobId = "76767676-7676-4767-8767-767676767676";
  await page.route(`**/api/v1/store/datasets/${datasetId}/cleaning-jobs`, async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(cleaningJob(datasetId, cleaningJobId, "running")) });
  });
  await page.route(`**/api/v1/store/datasets/${datasetId}/cleaning-jobs/${cleaningJobId}`, async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(cleaningJob(datasetId, cleaningJobId, cleaningReleased ? "completed" : "running")) });
  });
  await page.route(`**/api/v1/store/datasets/${datasetId}/cleaning-jobs/${cleaningJobId}/events*`, async (route) => {
    await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "test polling fallback" }) });
  });
  await page.route(`**/api/v1/store/datasets/${datasetId}/cleaning-jobs/${cleaningJobId}/result`, async (route) => {
    await cleaningGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        dataset_id: datasetId,
        run_id: "77777777-7777-4777-8777-777777777777",
        version: 1,
        provider: "local",
        model: "rules",
        source: "rules",
        raw_row_count: 2,
        cleaned_row_count: 2,
        cleaned_column_count: 2,
        result_markdown: "Cleaning complete.",
        preview_records: [{ customer_id: 1, value: 10 }, { customer_id: 2, value: 20 }],
        warnings: [],
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_cleaning_navigation");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "数据集" }).click();

  await page.locator('input[type="file"][accept=".csv,.xlsx,.json,.txt"]').setInputFiles({
    name: "navigation-cleaning.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("customer_id,value\n1,10\n2,20\n"),
  });
  await page.getByRole("button", { name: /导入并创建清洗任务/ }).click();
  await expect(page.getByText("已导入 2 行，正在清洗...", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "首页" }).click();
  await expect(page.getByRole("heading", { name: "工作区" })).toBeVisible();
  await page.getByRole("button", { name: "数据集" }).click();
  await expect(page.getByText("navigation-cleaning.csv", { exact: true })).toBeVisible();
  await expect(page.getByText("已导入 2 行，正在清洗...", { exact: true })).toBeVisible();

  releaseCleaning();
  await expect(page.getByText("完成：导入 2 行并创建清洗版本。", { exact: true })).toBeVisible();
  await expect(page.getByText(/批量处理完成：本次处理 1 个，成功 1 个，失败 0 个/)).toBeVisible();
});

test("multi-file import automatically configures and saves relationships", async ({ page }) => {
  const firstId = "abababab-abab-4bab-8bab-abababababab";
  const secondId = "cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd";
  const groupId = "efefefef-efef-4fef-8fef-efefefefefef";
  let importCount = 0;
  let releaseRelationships!: () => void;
  const relationshipGate = new Promise<void>((resolve) => {
    releaseRelationships = resolve;
  });
  const datasets = [
    { dataset_id: firstId, name: "orders.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
    { dataset_id: secondId, name: "customers.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
  ];
  const relationship = {
    left_dataset_id: firstId,
    right_dataset_id: secondId,
    left_column: "customer_id",
    right_column: "customer_id",
    join_type: "left",
    enabled: true,
    confidence: 0.96,
    source: "rules",
    reason: "字段同名并通过样本校验。",
    relationship_type: "many_to_one",
    risk_note: "",
  };
  const buildGroup = (relationships: typeof relationship[] = []) => ({
    group_id: groupId,
    name: "自动关系数据包",
    description: "批量上传文件自动创建",
    tables: [
      { dataset: datasets[0], row_count: 2, column_count: 3, columns: ["order_id", "customer_id", "amount"], entity_type: "fact", sample_records: [] },
      { dataset: datasets[1], row_count: 2, column_count: 2, columns: ["customer_id", "segment"], entity_type: "dimension", sample_records: [] },
    ],
    relationships,
    metadata: {},
  });

  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets: [] }) });
  });
  await page.route("**/api/v1/store/files/import", async (route) => {
    const dataset = datasets[importCount++];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ dataset, inserted: 2, preview_records: [{ customer_id: "C1" }, { customer_id: "C2" }] }),
    });
  });
  await page.route("**/api/v1/store/datasets/*/cleaning-jobs", async (route) => {
    const datasetId = route.request().url().includes(firstId) ? firstId : secondId;
    const jobId = datasetId === firstId ? "11111111-2222-4333-8444-555555555555" : "22222222-3333-4444-8555-666666666666";
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(cleaningJob(datasetId, jobId, "completed")) });
  });
  await page.route("**/api/v1/store/datasets/*/cleaning-jobs/*/result", async (route) => {
    const datasetId = route.request().url().includes(firstId) ? firstId : secondId;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        dataset_id: datasetId,
        run_id: crypto.randomUUID(),
        version: 1,
        provider: "local",
        model: "rules",
        source: "rules",
        raw_row_count: 2,
        cleaned_row_count: 2,
        cleaned_column_count: 2,
        result_markdown: "Cleaning complete.",
        preview_records: [{ customer_id: "C1" }, { customer_id: "C2" }],
        warnings: [],
      }),
    });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(buildGroup()) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [] }) });
  });
  await page.route(`**/api/v1/store/dataset-groups/${groupId}/relationships/auto-configure`, async (route) => {
    await relationshipGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        group: buildGroup([relationship]),
        candidates: [relationship],
        saved_relationships: [relationship],
        primary_dataset_id: firstId,
        unresolved_dataset_ids: [],
        llm_used: true,
        compact_context: {},
        validation_issues: [],
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_auto_relationship_import");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "数据集" }).click();
  await page.locator('input[type="file"][accept=".csv,.xlsx,.json,.txt"]').setInputFiles([
    { name: "orders.csv", mimeType: "text/csv", buffer: Buffer.from("order_id,customer_id,amount\nO1,C1,10\nO2,C2,20\n") },
    { name: "customers.csv", mimeType: "text/csv", buffer: Buffer.from("customer_id,segment\nC1,A\nC2,B\n") },
  ]);
  await page.getByRole("button", { name: /导入并创建清洗任务/ }).click();

  const pipeline = page.locator(".dataset-import-pipeline");
  await expect(pipeline).toContainText("正在自动建立数据关系");
  await expect(pipeline).toContainText(/已等待 \d+ 秒/);
  await expect(page.getByRole("button", { name: "正在自动识别关系" })).toBeDisabled();

  releaseRelationships();
  await expect(pipeline).toContainText("数据已准备完成");
  await expect(pipeline).toContainText("1 条自动关系");
  await expect(pipeline).toContainText("已使用语义补充");
  const completionStatus = page.getByRole("status").filter({ hasText: /批量处理完成：本次处理 2 个，成功 2 个，失败 0 个/ });
  await expect(completionStatus).toBeVisible();
  await expect(page.getByRole("alert").filter({ hasText: /失败 0 个/ })).toHaveCount(0);
});

test("an unconfirmed dataset group leads directly to relationship management", async ({ page }) => {
  const ordersId = "88888888-8888-4888-8888-888888888888";
  const customersId = "99999999-9999-4999-8999-999999999999";
  const groupId = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
  const datasets = [
    {
      dataset_id: ordersId,
      name: "orders.csv",
      source_type: "csv",
      status: "cleaned",
      source_metadata: {},
      created_at: "2026-07-11T10:00:00Z",
    },
    {
      dataset_id: customersId,
      name: "customers.csv",
      source_type: "csv",
      status: "cleaned",
      source_metadata: {},
      created_at: "2026-07-11T10:00:00Z",
    },
  ];
  const group = {
    group_id: groupId,
    name: "电商订单数据包",
    description: "批量上传文件自动创建",
    tables: [
      { dataset: datasets[0], row_count: 1000, column_count: 3, columns: ["order_id", "customer_id", "amount"], entity_type: "fact", sample_records: [] },
      { dataset: datasets[1], row_count: 200, column_count: 2, columns: ["customer_id", "segment"], entity_type: "dimension", sample_records: [] },
    ],
    relationships: [],
    metadata: {},
    created_at: "2026-07-11T10:00:00Z",
  };

  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [group] }) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_relationship_guidance");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "分析任务" }).click();

  await page.locator(".analysis-form-grid select").first().selectOption(groupId);
  await expect(page.getByRole("heading", { name: "数据包关系尚未自动建立" })).toBeVisible();
  const confirmationSteps = page.locator(".analysis-relationship-steps");
  await expect(confirmationSteps).toContainText("运行自动识别");
  await expect(confirmationSteps).toContainText("规则与语义联合判断");
  await expect(confirmationSteps).toContainText("校验后自动保存");
  await expect(page.getByRole("button", { name: "等待关系识别" })).toBeDisabled();

  await page.getByRole("button", { name: /前往自动识别/ }).click();
  await expect(page.getByRole("heading", { name: "导入工作台" })).toBeVisible();
  await expect(page.getByRole("tab", { name: /关系管理/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("尚未建立可靠关系", { exact: true })).toBeVisible();
  await expect(page.getByText("点击“自动识别关系”，系统会完成推荐、校验与保存。", { exact: true })).toBeVisible();
});

test("relationship recommendation communicates progress and saved readiness", async ({ page }) => {
  const ordersId = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
  const customersId = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
  const groupId = "dddddddd-dddd-4ddd-8ddd-dddddddddddd";
  const datasets = [
    { dataset_id: ordersId, name: "orders.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
    { dataset_id: customersId, name: "customers.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
  ];
  const candidate = {
    left_dataset_id: ordersId,
    right_dataset_id: customersId,
    left_column: "customer_id",
    right_column: "customer_id",
    join_type: "left",
    enabled: true,
    confidence: 0.96,
    source: "rules",
    estimated_match_rate: 0.98,
    reason: "字段同名、角色兼容，且样本匹配率高。",
    relationship_type: "many_to_one",
    risk_note: "确认附表键唯一后再汇总。",
  };
  let relationships: typeof candidate[] = [];
  const buildGroup = () => ({
    group_id: groupId,
    name: "电商订单数据包",
    description: "批量上传文件自动创建",
    tables: [
      { dataset: datasets[0], row_count: 1000, column_count: 3, columns: ["order_id", "customer_id", "amount"], entity_type: "fact", sample_records: [] },
      { dataset: datasets[1], row_count: 200, column_count: 2, columns: ["customer_id", "segment"], entity_type: "dimension", sample_records: [] },
    ],
    relationships,
    metadata: {},
  });
  let releaseSuggestions!: () => void;
  const suggestionGate = new Promise<void>((resolve) => {
    releaseSuggestions = resolve;
  });

  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [buildGroup()] }) });
  });
  await page.route(`**/api/v1/store/dataset-groups/${groupId}/relationships/auto-configure`, async (route) => {
    await suggestionGate;
    relationships = [candidate];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        group: buildGroup(),
        candidates: [candidate],
        saved_relationships: [candidate],
        primary_dataset_id: ordersId,
        unresolved_dataset_ids: [],
        llm_used: true,
        compact_context: {},
        validation_issues: [],
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_relationship_progress");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "数据集" }).click();
  await page.getByRole("tab", { name: /关系管理/ }).click();

  await page.getByRole("button", { name: "自动识别关系" }).click();
  await expect(page.getByRole("status")).toContainText("正在识别并保存可靠关系");
  await expect(page.getByRole("status")).toContainText(/已等待 \d+ 秒/);
  await expect(page.getByLabel("关系推荐处理内容")).toContainText("规则候选");
  await expect(page.getByLabel("关系推荐处理内容")).toContainText("语义补充");
  await expect(page.getByLabel("关系推荐处理内容")).toContainText("样本匹配校验");

  releaseSuggestions();
  await expect(page.getByText("字段同名、角色兼容，且样本匹配率高。", { exact: true })).toBeVisible();
  await expect(page.getByText("已自动采用", { exact: true })).toBeVisible();
  await expect(page.getByText("数据包已就绪", { exact: true })).toBeVisible();
  await expect(page.getByText("已自动校验并保存 1 条关系。", { exact: true })).toBeVisible();
});

test("dataset group analysis uses the saved relationship primary table", async ({ page }) => {
  const sellersId = "10101010-1010-4010-8010-101010101010";
  const productsId = "20202020-2020-4020-8020-202020202020";
  const translationId = "30303030-3030-4030-8030-303030303030";
  const groupId = "40404040-4040-4040-8040-404040404040";
  const datasets = [
    { dataset_id: sellersId, name: "sellers.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
    { dataset_id: productsId, name: "products.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
    { dataset_id: translationId, name: "translation.csv", source_type: "csv", status: "cleaned", source_metadata: {} },
  ];
  const relationship = {
    left_dataset_id: productsId,
    right_dataset_id: translationId,
    left_column: "product_category_name",
    right_column: "product_category_name",
    join_type: "left",
    enabled: true,
    confidence: 0.92,
    source: "rules",
    reason: "Validated relationship.",
    relationship_type: "many_to_one",
    risk_note: "",
  };
  const group = {
    group_id: groupId,
    name: "Mixed commerce package",
    description: "Three imported tables",
    tables: datasets.map((dataset) => ({ dataset, row_count: 10, column_count: 2, columns: ["id", "name"], entity_type: "unknown", sample_records: [] })),
    relationships: [relationship],
    metadata: {},
  };
  let submittedBody: Record<string, unknown> | null = null;

  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [group] }) });
  });
  await page.route("**/api/v1/analysis/jobs*", async (route) => {
    if (route.request().method() === "POST") {
      submittedBody = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          job_id: "50505050-5050-4050-8050-505050505050",
          dataset_id: productsId,
          question: "分析产品品类表现",
          status: "queued",
          progress: 0,
          current_stage: "queued",
          events: [],
          created_at: "2026-07-11T12:00:00Z",
          updated_at: "2026-07-11T12:00:00Z",
        }),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_group_primary_alignment");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "分析任务" }).click();
  await page.locator(".analysis-form-grid select").first().selectOption(groupId);

  const primarySelect = page.locator(".analysis-form-grid select").nth(1);
  await expect(primarySelect).toBeDisabled();
  await expect(primarySelect).toHaveValue(productsId);
  await page.getByLabel("分析问题").fill("分析产品品类表现");
  await page.getByRole("button", { name: "开始分析" }).click();
  await expect.poll(() => submittedBody).not.toBeNull();
  expect(submittedBody?.dataset_id).toBe(productsId);
  expect(submittedBody?.relationship_plan).toEqual([
    {
      left_dataset_id: productsId,
      right_dataset_id: translationId,
      left_column: "product_category_name",
      right_column: "product_category_name",
      join_type: "left",
    },
  ]);
});

test("low-confidence semantic plan requires explicit confirmation", async ({ page }) => {
  const datasetId = "99999999-9999-4999-8999-999999999999";
  await page.route("**/api/v1/store/datasets", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets: [{ dataset_id: datasetId, name: "orders", source_type: "csv", status: "cleaned" }] }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [] }) });
  });
  await page.route("**/api/v1/analysis/jobs?*", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });
  await page.route("**/api/v1/analysis/plans", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        decision_id: "88888888-8888-4888-8888-888888888888",
        semantic_source: "published",
        semantic_model_id: "77777777-7777-4777-8777-777777777777",
        semantic_model_version: 1,
        semantic_plan: { metric_ids: [], dimension_ids: [], ambiguities: ["Ambiguous metric"] },
        confidence_breakdown: { intent: 0.65, metric: 0.15, dimension: null, time: null, join: null, data_quality: 0.82, route: 0.45 },
        raw_confidence: 0.44,
        calibrated_confidence: 0.41,
        confidence_level: "low",
        requires_confirmation: true,
        ambiguities: ["Ambiguous metric"],
        evidence: ["No metric matched"],
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_semantic_confirmation");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "分析任务" }).click();
  await page.getByLabel("分析问题").fill("分析客户价值");
  await page.getByRole("button", { name: "开始分析" }).click();
  await expect(page.getByText("语义计划 · LOW")).toBeVisible();
  await expect(page.getByLabel("我已确认该指标、维度和 Join 口径")).not.toBeChecked();
});

test("default autonomous analysis submits loop mode and restores repair trace", async ({ page }, testInfo) => {
  const datasetId = "12121212-1212-4212-8212-121212121212";
  const jobId = "13131313-1313-4313-8313-131313131313";
  let submittedBody: Record<string, unknown> | null = null;
  const completedJob = {
    job_id: jobId,
    dataset_id: datasetId,
    question: "按区域分析销售额",
    status: "completed",
    progress: 100,
    current_stage: "complete",
    agent_mode: "loop",
    loop_terminal_reason: "model_finished",
    loop_summary: { iterations: 2, decisions: 3, tool_calls: 2, failed_tools: 1 },
    events: [
      { sequence: 1, stage: "loop_decide", progress: 28, message: "Selected tool: execute_safe_sql", status: "completed", event_type: "decision", iteration: 1, tool_name: "execute_safe_sql", payload: { remaining_budget: { tool_calls_remaining: 11, decisions_remaining: 15, tokens_remaining: 49000 } }, created_at: "2026-07-13T08:00:01Z" },
      { sequence: 2, stage: "loop_execute", progress: 31, message: "execute_safe_sql failed.", status: "failed", event_type: "tool_execution", iteration: 1, tool_name: "execute_safe_sql", payload: { error_type: "sql_error" }, created_at: "2026-07-13T08:00:02Z" },
      { sequence: 3, stage: "loop_repair", progress: 31, message: "Repairing sql_error failure; the next decision must change arguments or tool.", status: "running", event_type: "repair", iteration: 1, tool_name: "execute_safe_sql", payload: { error_type: "sql_error" }, created_at: "2026-07-13T08:00:03Z" },
      { sequence: 4, stage: "loop_execute", progress: 34, message: "execute_safe_sql succeeded.", status: "completed", event_type: "tool_execution", iteration: 2, tool_name: "execute_safe_sql", payload: { result_summary: "Produced 2 bounded row(s)." }, created_at: "2026-07-13T08:00:04Z" },
      { sequence: 5, stage: "loop_finalize", progress: 70, message: "Autonomous loop finalized: model_finished.", status: "completed", event_type: "loop_finalize", iteration: 2, payload: {}, created_at: "2026-07-13T08:00:05Z" },
      { sequence: 6, stage: "complete", progress: 100, message: "Analysis job completed.", status: "completed", created_at: "2026-07-13T08:00:06Z" },
    ],
    created_at: "2026-07-13T08:00:00Z",
    updated_at: "2026-07-13T08:00:06Z",
    completed_at: "2026-07-13T08:00:06Z",
  };

  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets: [{ dataset_id: datasetId, name: "中文销售.txt", source_type: "txt", status: "cleaned", source_metadata: {} }] }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [] }) });
  });
  await page.route("**/api/v1/analysis/plans", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ decision_id: "14141414-1414-4414-8414-141414141414", semantic_source: "legacy", semantic_plan: {}, confidence_breakdown: {}, raw_confidence: 0.6, calibrated_confidence: 0.6, confidence_level: "medium", requires_confirmation: false, ambiguities: [], evidence: [] }) });
  });
  await page.route("**/api/v1/analysis/jobs**", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "POST" && url.pathname.endsWith("/analysis/jobs")) {
      submittedBody = route.request().postDataJSON();
      await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ ...completedJob, status: "queued", progress: 0, current_stage: "queued", events: [] }) });
      return;
    }
    if (url.pathname.endsWith(`/analysis/jobs/${jobId}/result`)) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ dataset_id: datasetId, question: completedJob.question, report_markdown: "# Loop report", agent_mode: "loop", loop_summary: completedJob.loop_summary, loop_terminal_reason: "model_finished", workflow_trace: [], multimodal_inputs: [], multi_dataset_context: null, planner_metadata: null }) });
      return;
    }
    if (url.pathname.endsWith(`/analysis/jobs/${jobId}`)) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(completedJob) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_agent_loop");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "分析任务" }).click();
  await page.getByLabel("分析问题").fill(completedJob.question);
  const autonomousMode = page.getByRole("radio", { name: /自主分析/ });
  const compatibilityMode = page.getByRole("radio", { name: "兼容模式" });
  await expect(autonomousMode).toHaveAttribute("aria-checked", "true");
  await compatibilityMode.click();
  await expect(compatibilityMode).toHaveAttribute("aria-checked", "true");
  await expect(page.getByText("兼容分析流程", { exact: true })).toBeVisible();
  await autonomousMode.click();
  await expect(autonomousMode).toHaveAttribute("aria-checked", "true");
  await page.screenshot({ path: path.join(os.tmpdir(), `datamind-analysis-mode-${testInfo.project.name}.png`), fullPage: false });
  await page.getByRole("button", { name: "开始分析" }).click();

  await expect.poll(() => submittedBody).not.toBeNull();
  expect(submittedBody?.agent_mode).toBe("loop");
  const loopTrace = page.getByRole("region", { name: "自主分析循环轨迹" });
  await expect(loopTrace).toBeVisible();
  await expect(loopTrace.getByText("自动修复", { exact: true })).toBeVisible();
  await expect(loopTrace.getByText("execute_safe_sql").first()).toBeVisible();
  await expect(loopTrace.getByText("证据充分", { exact: true })).toBeVisible();
  await page.screenshot({ path: path.join(os.tmpdir(), `datamind-agent-loop-${testInfo.project.name}.png`), fullPage: false });
});

test("report loop restores validation repair and idempotent commit trace", async ({ page }) => {
  const datasetId = "15151515-1515-4515-8515-151515151515";
  const jobId = "16161616-1616-4616-8616-161616161616";
  const job = {
    job_id: jobId,
    dataset_id: datasetId,
    question: "生成区域销售报告",
    status: "completed",
    progress: 100,
    current_stage: "complete",
    agent_mode: "loop",
    loop_terminal_reason: "model_finished",
    report_strategy: "llm",
    report_revision_count: 2,
    report_terminal_reason: "validated",
    loop_summary: { tool_calls: 1, report: { strategy: "llm", revision_count: 2, terminal_reason: "validated" } },
    events: [
      { sequence: 1, stage: "report_decide", progress: 95, message: "Report strategy selected: llm.", status: "completed", event_type: "report_decision", iteration: 1, payload: { strategy: "llm" }, created_at: "2026-07-13T09:00:01Z" },
      { sequence: 2, stage: "report_execute", progress: 96, message: "Report draft revision 1 generated.", status: "completed", event_type: "report_draft", iteration: 1, payload: {}, created_at: "2026-07-13T09:00:02Z" },
      { sequence: 3, stage: "report_verify", progress: 97, message: "Report validation outcome: report_issue.", status: "failed", event_type: "report_validation", iteration: 1, payload: { unsupported_numeric_findings: ["区域销售额"] }, created_at: "2026-07-13T09:00:03Z" },
      { sequence: 4, stage: "report_repair", progress: 97, message: "Report draft will be regenerated with validation feedback.", status: "completed", event_type: "report_repair", iteration: 1, payload: {}, created_at: "2026-07-13T09:00:04Z" },
      { sequence: 5, stage: "report_verify", progress: 97, message: "Report validation outcome: sufficient.", status: "completed", event_type: "report_validation", iteration: 2, payload: {}, created_at: "2026-07-13T09:00:05Z" },
      { sequence: 6, stage: "report_commit", progress: 99, message: "Validated report committed idempotently.", status: "completed", event_type: "report_commit", iteration: 2, payload: {}, created_at: "2026-07-13T09:00:06Z" },
      { sequence: 7, stage: "complete", progress: 100, message: "Analysis job completed.", status: "completed", created_at: "2026-07-13T09:00:07Z" },
    ],
    created_at: "2026-07-13T09:00:00Z",
    updated_at: "2026-07-13T09:00:07Z",
    completed_at: "2026-07-13T09:00:07Z",
  };
  await page.route("**/api/v1/store/datasets", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ datasets: [{ dataset_id: datasetId, name: "区域销售.csv", source_type: "csv", status: "cleaned", source_metadata: {} }] }) });
  });
  await page.route("**/api/v1/store/dataset-groups", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ groups: [] }) });
  });
  await page.route("**/api/v1/analysis/plans", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ decision_id: "17171717-1717-4717-8717-171717171717", semantic_source: "legacy", semantic_plan: {}, confidence_breakdown: {}, raw_confidence: 0.65, calibrated_confidence: 0.65, confidence_level: "medium", requires_confirmation: false, ambiguities: [], evidence: [] }) });
  });
  await page.route("**/api/v1/analysis/jobs**", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "POST" && url.pathname.endsWith("/analysis/jobs")) {
      await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ ...job, status: "queued", progress: 0, current_stage: "queued", events: [] }) });
      return;
    }
    if (url.pathname.endsWith(`/analysis/jobs/${jobId}/result`)) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ dataset_id: datasetId, question: job.question, report_markdown: "# Report loop", agent_mode: "loop", loop_summary: job.loop_summary, loop_terminal_reason: "model_finished", report_strategy: "llm", report_revision_count: 2, report_terminal_reason: "validated", workflow_trace: [], multimodal_inputs: [], multi_dataset_context: null, planner_metadata: null }) });
      return;
    }
    if (url.pathname.endsWith(`/analysis/jobs/${jobId}`)) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(job) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });

  await page.goto("/");
  await page.getByLabel("用户名").fill("qa_report_loop");
  await page.getByLabel("密码").fill("qa-reliability-password");
  await page.getByRole("button", { name: /Log in|登录/ }).click();
  await page.getByRole("button", { name: "分析任务" }).click();
  await page.getByLabel("分析问题").fill(job.question);
  await expect(page.getByRole("radio", { name: /自主分析/ })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("button", { name: "开始分析" }).click();

  const reportLoop = page.getByRole("region", { name: "报告生成循环轨迹" });
  await expect(reportLoop).toBeVisible();
  await expect(reportLoop.getByText("报告修订", { exact: true })).toBeVisible();
  await expect(reportLoop.getByText("提交报告", { exact: true })).toBeVisible();
  await expect(reportLoop.getByText("validated", { exact: true })).toBeVisible();
});
